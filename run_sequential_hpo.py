"""Sequential (coordinate-descent) hyperparameter tuning with parallel GPU support.

Usage:
    python run_sequential_hpo.py --config config_files/models/sibrar_onion_e --gpus 0 1
    python run_sequential_hpo.py --config config_files/models/sibrar_onion_e --gpus 0 1 \\
        --checkpoint checkpoints/my_run.json   # resume after interruption
"""

import argparse
import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from os import path
from types import SimpleNamespace

import numpy as np

from utils import (load_yaml, save_yaml, merge_dicts,
                    is_oom_message, is_permanent_failure)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _is_oom(exc: Exception) -> bool:
    return is_oom_message(str(exc))


def _is_permanent_failure(err_msg) -> bool:
    return is_permanent_failure(err_msg)


def _clear_gpu():
    if _HAS_TORCH:
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _scalar(v):
    return v[0] if isinstance(v, list) else v


def _coerce(v):
    """PyYAML 6 parses '1e-6' as a str, not float."""
    from ast import literal_eval
    if isinstance(v, str):
        try:
            result = literal_eval(v)
            if isinstance(result, (int, float, bool)):
                return result
            if isinstance(result, list):
                return [_coerce(x) for x in result]
        except (ValueError, SyntaxError):
            pass
        return v
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


def _tunable(model_cfg: dict) -> dict:
    return {
        k: _coerce(v) for k, v in model_cfg.items()
        if isinstance(v, list) and len(v) > 1 and k != "meta"
    }


def _build_params_ns(model_cfg: dict, overrides: dict) -> SimpleNamespace:
    flat: dict = {}
    for k, v in model_cfg.items():
        if k == "meta":
            meta_dict = dict(v) if isinstance(v, dict) else {}
            meta_dict.update({"save_recs": False, "save_weights": False, "verbose": False})
            flat[k] = SimpleNamespace(**meta_dict)
        elif isinstance(v, list):
            flat[k] = _coerce(v[0])
        else:
            flat[k] = v
    flat["log_wandb"] = False
    flat.update({k: _coerce(v) for k, v in overrides.items()})
    return SimpleNamespace(**flat)


def finalize_output_meta(config: dict, model_key: str) -> dict:
    """Turn on recs/weights persistence in the tuned config the sweep emits."""
    out = copy.deepcopy(config)
    model_cfg = out.get("experiment", {}).get("models", {}).get(model_key)
    if model_cfg is None:
        return out
    meta = dict(model_cfg.get("meta") or {})
    meta["save_recs"] = True
    meta["save_weights"] = True
    model_cfg["meta"] = meta
    return out


def set_training_epochs(config: dict, model_key: str, epochs: int) -> dict:
    """Give the emitted tuned config the full training epoch budget."""
    out = copy.deepcopy(config)
    model_cfg = out.get("experiment", {}).get("models", {}).get(model_key)
    if model_cfg is None:
        return out
    model_cfg["epochs"] = epochs
    return out


def _load_model_class(key: str, base_ns: SimpleNamespace):
    if key.startswith("external."):
        spec = importlib.util.spec_from_file_location(
            "external", path.relpath(base_ns.external_models_path)
        )
        external = importlib.util.module_from_spec(spec)
        external.backend = base_ns.backend
        sys.modules[spec.name] = external
        spec.loader.exec_module(external)
        return getattr(importlib.import_module("external"), key.split(".", 1)[1])
    return getattr(importlib.import_module("elliot.recommender"), key)


def _best_ndcg5(model) -> float:
    """Extract the best validation nDCG@5 across all training epochs."""
    return max(r[5]['val_results']['nDCG'] for r in model._results)


def _run_trial(model_class, data_objs, base_ns: SimpleNamespace,
               params_ns: SimpleNamespace, fold_idx: int):
    """Train in-process. Returns (avg_ndcg5, results) or (-inf, None) on OOM."""
    scores, results = [], []
    try:
        for data_obj in data_objs:
            model = model_class(data=data_obj, config=base_ns, params=params_ns)
            model._fold_index = fold_idx
            model.train()
            scores.append(_best_ndcg5(model))
            results.append(model.get_results())
            del model
            _clear_gpu()
        return float(np.mean(scores)), results
    except Exception as exc:
        if _is_oom(exc):
            print(f"  [OOM] {exc}")
            _clear_gpu()
            return float("-inf"), None
        raise


def _build_config(args) -> dict:
    base_config = load_yaml("config_files/DEFAULT.yml")
    extended_config = load_yaml(args.config + ".yml")
    config = merge_dicts(base_config, extended_config)
    model_defaults = load_yaml("config_files/models/model_defaults.yml")
    for key in list(config["experiment"]["models"]):
        config["experiment"]["models"][key] = {
            **model_defaults,
            **config["experiment"]["models"][key],
        }
        config["experiment"]["models"][key]["seed"] = args.seed
        if "modalities" in config:
            config["experiment"]["models"][key]["modalities"] = config["modalities"]
        if "loaders" in config:
            config["experiment"]["models"][key]["loaders"] = config["loaders"]
    return config


def _setup_elliot(config: dict, gpu_id: int, tmp_suffix: str):
    """Write temp config, init Elliot, load + return dataset."""
    config["experiment"]["gpu"] = gpu_id  # NameSpaceModel reads this for CUDA_VISIBLE_DEVICES
    tmp_path = f"config_files/tmp_seq_{tmp_suffix}.yml"
    save_yaml(config, tmp_path)

    try:
        import elliot.run as _em
        elliot_here = path.abspath(path.dirname(_em.__file__))

        from elliot.namespace.namespace_model_builder import NameSpaceBuilder
        from elliot.utils import logging as logging_project

        builder = NameSpaceBuilder(tmp_path, elliot_here, path.abspath(path.dirname(tmp_path)))
        base = builder.base
        base_ns = base.base_namespace
        logging_project.init(base_ns.path_logger_config, base_ns.path_log_folder)
        # Replicate the init step from run_experiment that models rely on
        base_ns.evaluation.relevance_threshold = getattr(
            base_ns.evaluation, "relevance_threshold", 0
        )

        dataloader_class = getattr(
            importlib.import_module("elliot.dataset"), base_ns.data_config.dataloader
        )
        data_test_list = dataloader_class(config=base_ns).generate_dataobjects()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return base_ns, data_test_list


def _load_ckpt(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def _save_ckpt(ckpt_path: str, data: dict):
    os.makedirs(path.dirname(path.abspath(ckpt_path)), exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [checkpoint] → {ckpt_path}")


def _run_candidates_parallel(candidates, param, best_params, raw_model_cfg,
                              model_key, args, gpus, indices=None) -> tuple:
    """Spawn one subprocess per candidate, scheduling across available GPUs."""
    max_per_gpu = max(1, getattr(args, "max_per_gpu", 2))
    scores = [float("-inf")] * len(candidates)
    errors = [None] * len(candidates)
    config_tag = args.config.replace("/", "_").replace("config_files_", "")
    run_positions = list(range(len(candidates))) if indices is None else list(indices)

    # idx -> set of GPUs already attempted for this candidate (OOM'd on them)
    tried_gpus = {idx: set() for idx in run_positions}
    pending = list(run_positions)  # candidates still needing a slot
    # candidates that OOM'd while packed and are waiting specifically for an
    # idle GPU (not just any under-capacity one) before their next attempt
    wants_solo = set()
    # slots: up to len(gpus) * max_per_gpu concurrent subprocesses, at most
    # max_per_gpu per GPU.
    running = {}  # slot_id -> (idx, val, gpu, proc, tmp_result, stderr_file)
    next_slot_id = [0]

    def _gpu_load():
        """gpu -> number of candidates currently running on it."""
        load = {g: 0 for g in gpus}
        for _, _, g, _, _, _ in running.values():
            load[g] += 1
        return load

    def _launch(idx, gpu):
        val = candidates[idx]
        attempt = len(tried_gpus[idx])
        slot_id = next_slot_id[0]
        next_slot_id[0] += 1
        tmp_result = f"config_files/tmp_worker_{config_tag}_{param}_{idx}_{attempt}.json"
        cmd = [
            sys.executable, __file__,
            "--config", args.config,
            "--seed", str(args.seed),
            "--tune_epochs", str(args.tune_epochs),
            "--tune_fold", str(args.tune_fold),
            "--worker",
            "--worker_gpu", str(gpu),
            "--worker_param", param,
            "--worker_value", json.dumps(val),
            "--worker_best_params", json.dumps(best_params),
            "--worker_model_key", model_key,
            "--worker_tmp_file", tmp_result,
        ]
        stderr_file = tmp_result.replace(".json", ".err")
        load_before = _gpu_load()[gpu]
        share_tag = f" (sharing GPU with {load_before} other job(s))" if load_before else ""
        retry_tag = f" (retry {attempt}, prior GPU(s) {sorted(tried_gpus[idx])} OOM'd)" if attempt else ""
        print(f"    GPU {gpu}: {param}={val!r}{share_tag}{retry_tag}", flush=True)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL,
            stderr=open(stderr_file, "w"),
        )
        tried_gpus[idx].add(gpu)
        wants_solo.discard(idx)
        running[slot_id] = (idx, val, gpu, proc, tmp_result, stderr_file)

    def _pick_gpu_for(idx, prefer_solo):
        """Pick a GPU for a retry."""
        untried = [g for g in gpus if g not in tried_gpus[idx]]
        load = _gpu_load()
        idle_untried = [g for g in untried if load[g] == 0]
        if idle_untried:
            return idle_untried[0]
        if prefer_solo:
            return None
        under_cap_untried = [g for g in untried if load[g] < max_per_gpu]
        if under_cap_untried:
            return under_cap_untried[0]
        if untried:
            return untried[0]
        return None

    def _fill_idle_slots():
        while pending:
            load = _gpu_load()
            available_gpus = [g for g in gpus if load[g] < max_per_gpu]
            if not available_gpus:
                break
            # Prefer emptiest GPUs first, and a candidate that hasn't already
            # tried that GPU.
            available_gpus.sort(key=lambda g: load[g])
            launched = False
            for gpu in available_gpus:
                is_idle = load[gpu] == 0
                candidates_here = [idx for idx in pending if gpu not in tried_gpus[idx]]
                if not is_idle:
                    # Busy GPU: only offer it to candidates that don't insist
                    # on running solo (they OOM'd while packed already).
                    candidates_here = [idx for idx in candidates_here if idx not in wants_solo]
                else:
                    # Idle GPU: give solo-wanting candidates priority so they
                    # aren't starved by ordinary pending candidates.
                    candidates_here.sort(key=lambda idx: idx not in wants_solo)
                choice = candidates_here[0] if candidates_here else None
                if choice is None:
                    continue
                pending.remove(choice)
                _launch(choice, gpu)
                launched = True
                if not pending:
                    break
                load = _gpu_load()
                if all(load[g] >= max_per_gpu for g in gpus):
                    break
            if not launched:
                break

    _fill_idle_slots()

    while running:
        slot_id = None
        while slot_id is None:
            for sid, (_, _, _, proc, _, _) in running.items():
                if proc.poll() is not None:
                    slot_id = sid
                    break
            if slot_id is None:
                time.sleep(0.5)
        idx, val, gpu, proc, tmp_result, stderr_file = running.pop(slot_id)
        was_packed = _gpu_load()[gpu] > 0  # other job(s) still on this GPU

        is_oom = False
        if proc.returncode == 0 and os.path.exists(tmp_result):
            try:
                with open(tmp_result) as f:
                    res = json.load(f)
                if res.get("score") is None:
                    is_oom = True
                else:
                    scores[idx] = res["score"]
                os.remove(tmp_result)
            except Exception:
                scores[idx] = float("-inf")
        else:
            scores[idx] = float("-inf")
            err_msg = ""
            if os.path.exists(stderr_file):
                err_text = open(stderr_file).read().strip()
                # show/store last line only to avoid flooding the terminal
                last_line = err_text.splitlines()[-1] if err_text else ""
                errors[idx] = last_line or None
                err_msg = "  ← " + last_line if last_line else ""
            print(f"    GPU {gpu}: {param}={val!r} → crash (rc={proc.returncode}){err_msg}")
            if os.path.exists(tmp_result):
                os.remove(tmp_result)
        if os.path.exists(stderr_file):
            os.remove(stderr_file)

        if is_oom:
            next_gpu = _pick_gpu_for(idx, prefer_solo=was_packed)
            if next_gpu is not None:
                reason = "OOM while sharing a GPU, retrying solo" if was_packed else "OOM, retrying"
                print(f"    GPU {gpu}: {param}={val!r} → {reason} on GPU {next_gpu}")
                _launch(idx, next_gpu)
            elif was_packed and [g for g in gpus if g not in tried_gpus[idx]]:
                wants_solo.add(idx)
                pending.append(idx)
                print(f"    GPU {gpu}: {param}={val!r} → OOM while sharing, "
                      f"waiting for a free GPU to retry solo")
            else:
                scores[idx] = float("-inf")
                print(f"    GPU {gpu}: {param}={val!r} → OOM on all GPUs {gpus}")
        else:
            display = f"ndcg@5={scores[idx]:.5f}" if scores[idx] != float("-inf") else "crash/OOM"
            print(f"    GPU {gpu}: {param}={val!r} → {display}")

        _fill_idle_slots()

    return scores, errors


def worker_main(args):
    gpu = args.worker_gpu
    config = _build_config(args)
    config_tag = args.config.replace("/", "_").replace("config_files_", "")

    base_ns, data_test_list = _setup_elliot(config, gpu, f"worker_gpu{gpu}_{config_tag}")

    if args.tune_fold >= len(data_test_list):
        raise ValueError(f"tune_fold={args.tune_fold} but dataset has {len(data_test_list)} folds")
    tuning_data = data_test_list[args.tune_fold]

    from elliot.utils import logging as logging_project
    logging_project.prepare_logger(args.worker_model_key, base_ns.path_log_folder)

    raw_model_cfg = config["experiment"]["models"][args.worker_model_key]
    model_class = _load_model_class(args.worker_model_key, base_ns)

    best_params = json.loads(args.worker_best_params)
    val = json.loads(args.worker_value)
    overrides = dict(best_params)
    overrides[args.worker_param] = val
    overrides["epochs"] = args.tune_epochs
    params_ns = _build_params_ns(raw_model_cfg, overrides)

    score, _ = _run_trial(model_class, tuning_data, base_ns, params_ns, args.tune_fold)

    with open(args.worker_tmp_file, "w") as f:
        json.dump({
            "score": score if score != float("-inf") else None,
            "value": val,
            "param": args.worker_param,
        }, f)


def main(args):
    config = _build_config(args)
    config_tag = args.config.replace("/", "_").replace("config_files_", "")
    gpus = args.gpus

    # For single-GPU in-process mode, set CUDA_VISIBLE_DEVICES upfront
    if len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        base_ns, data_test_list = _setup_elliot(config, gpus[0], config_tag)
    else:
        # Multi-GPU: dataset is loaded fresh in each worker subprocess.
        # We still need base_ns for model-class loading and config inspection.
        base_ns, data_test_list = _setup_elliot(config, gpus[0], config_tag)

    for model_key in list(config["experiment"]["models"]):
        raw_model_cfg = config["experiment"]["models"][model_key]
        tunable = _tunable(raw_model_cfg)

        print(f"\n{'='*72}")
        print(f"Model: {model_key}")

        if not tunable:
            print("No tunable hyperparameters (all params have ≤1 candidate). Nothing to do.")
            continue

        print(f"Tunable params : {list(tunable.keys())}")
        print(f"Tune epochs    : {args.tune_epochs}  |  Tune fold: {args.tune_fold}")
        print(f"GPUs           : {gpus}")

        # Determine order
        if args.tune_order:
            order = [p for p in args.tune_order if p in tunable]
            extra = [p for p in tunable if p not in args.tune_order]
            if [p for p in args.tune_order if p not in tunable]:
                print(f"[warn] not in tunable set (ignored): "
                      f"{[p for p in args.tune_order if p not in tunable]}")
            if extra:
                print(f"[info] appending params not in --tune_order: {extra}")
                order += extra
        else:
            order = list(tunable.keys())
        print(f"Tuning order   : {order}")

        # Checkpoint
        safe_model = model_key.replace(".", "_")
        ckpt_path = args.checkpoint or f"checkpoints/{config_tag}_{safe_model}.json"
        print(f"Checkpoint     : {ckpt_path}")
        ckpt = _load_ckpt(ckpt_path)

        # Only track list-valued params in best_params.
        # _coerce converts PyYAML-6 string numerics ('1e-6') to proper floats.
        _list_keys = {k for k, v in raw_model_cfg.items()
                      if isinstance(v, list) and k != "meta"}
        _fresh = {k: _coerce(v[0]) for k, v in raw_model_cfg.items()
                  if k in _list_keys}
        # Filter checkpoint to known list-valued keys only, so stale entries
        # from old runs (e.g. modalities stored before the fix) are discarded.
        _ckpt_best = {k: v for k, v in ckpt.get("best_params", {}).items()
                      if k in _list_keys}
        best_params = _ckpt_best if _ckpt_best else _fresh
        completed = set(ckpt.get("completed_params", []))
        param_results = ckpt.get("param_results", {})

        if len(gpus) == 1:
            from elliot.utils import logging as logging_project
            logging_project.prepare_logger(model_key, base_ns.path_log_folder)
            model_class = _load_model_class(model_key, base_ns)
            tuning_data = data_test_list[args.tune_fold]

        # ── Sequential loop over params ───────────────────────────────────────
        for param in order:
            prev_result = param_results.get(param)

            if param in completed and not (args.retry_oom and prev_result):
                print(f"\n  [{param}] Already tuned → best={best_params.get(param)}. Skipping.")
                continue

            candidates = tunable[param]

            retry_indices = None
            if param in completed and args.retry_oom and prev_result:
                prev_scores = prev_result.get("scores", [])
                prev_errors = prev_result.get("errors", [None] * len(candidates))
                if len(prev_scores) != len(candidates):
                    print(f"\n  [{param}] Candidate list changed since last run "
                          f"({len(prev_scores)} → {len(candidates)}); re-running all.")
                    retry_indices = None
                else:
                    retry_indices = [
                        i for i, s in enumerate(prev_scores)
                        if s is None and not _is_permanent_failure(
                            prev_errors[i] if i < len(prev_errors) else None
                        )
                    ]
                    skipped = [
                        (candidates[i], prev_errors[i] if i < len(prev_errors) else None)
                        for i, s in enumerate(prev_scores)
                        if s is None and _is_permanent_failure(
                            prev_errors[i] if i < len(prev_errors) else None
                        )
                    ]
                    if not retry_indices:
                        print(f"\n  [{param}] Already tuned → best={best_params.get(param)}. "
                              f"No retriable OOM candidates. Skipping.")
                        for val, err in skipped:
                            print(f"    (permanent failure, not retried) {param}={val!r}  ← {err}")
                        continue
                    print(f"\n  [{param}] Retrying {len(retry_indices)} OOM/crashed candidate(s) "
                          f"of {len(candidates)}: {[candidates[i] for i in retry_indices]}")
                    for val, err in skipped:
                        print(f"    (permanent failure, not retried) {param}={val!r}  ← {err}")

            if retry_indices is None and param not in completed:
                print(f"\n  [{param}]  {len(candidates)} candidates: {candidates}")

            if len(gpus) == 1:
                # In-process sequential
                run_positions = list(range(len(candidates))) if retry_indices is None else retry_indices
                trial_scores = [float("-inf")] * len(candidates)
                trial_errors = [None] * len(candidates)
                for i in run_positions:
                    val = candidates[i]
                    overrides = dict(best_params)
                    overrides[param] = val
                    overrides["epochs"] = args.tune_epochs
                    params_ns = _build_params_ns(raw_model_cfg, overrides)
                    print(f"    {param}={val!r:30s}", end="", flush=True)
                    score, _ = _run_trial(model_class, tuning_data, base_ns, params_ns, args.tune_fold)
                    print(f"→ {'OOM' if score == float('-inf') else f'ndcg@5={score:.5f}'}")
                    trial_scores[i] = score
            else:
                # Parallel subprocesses, one per GPU
                trial_scores, trial_errors = _run_candidates_parallel(
                    candidates, param, best_params, raw_model_cfg, model_key, args, gpus,
                    indices=retry_indices,
                )

            # When retrying, candidates not re-run keep their previous score/error.
            if retry_indices is not None:
                prev_scores = prev_result.get("scores", [])
                prev_errors = prev_result.get("errors", [None] * len(candidates))
                for i in range(len(candidates)):
                    if i not in retry_indices:
                        trial_scores[i] = prev_scores[i] if prev_scores[i] is not None else float("-inf")
                        trial_errors[i] = prev_errors[i] if i < len(prev_errors) else None

            valid = [(s, c) for s, c in zip(trial_scores, candidates) if s != float("-inf")]
            if not valid:
                print(f"  [{param}] All candidates OOM. Keeping current: {best_params.get(param)}")
            else:
                best_score, best_val = max(valid, key=lambda x: x[0])
                best_params[param] = best_val
                print(f"  [{param}] Best: {best_val!r}  (ndcg@5={best_score:.5f})")

            param_results[param] = {
                "values_tried": candidates,
                "scores": [s if s != float("-inf") else None for s in trial_scores],
                "errors": trial_errors,
                "best_value": best_params.get(param),
                "best_ndcg5": max((s for s in trial_scores if s != float("-inf")), default=None),
            }
            completed.add(param)
            _save_ckpt(ckpt_path, {
                "model_key": model_key,
                "best_params": best_params,
                "completed_params": list(completed),
                "param_results": param_results,
            })

        # ── Summary + output config ───────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"Done: {model_key}  (tuned with {args.tune_epochs} epochs, fold {args.tune_fold})")
        for p in order:
            old = _scalar(raw_model_cfg.get(p, "?"))
            new = best_params.get(p, "?")
            tag = "  ← changed" if old != new else ""
            print(f"  {p}: {new!r}{tag}")

        out_cfg = copy.deepcopy(config)
        for k, v in best_params.items():
            if k in out_cfg["experiment"]["models"][model_key]:
                orig = out_cfg["experiment"]["models"][model_key][k]
                out_cfg["experiment"]["models"][model_key][k] = [v] if isinstance(orig, list) else v
        # Trials run with saving off; the tuned config feeds the final k-fold
        # runs, which exist to produce recs (and weights), so turn it back on.
        out_cfg = finalize_output_meta(out_cfg, model_key)
        # The tuning budget (small by design, in the input config and again via
        # --tune_epochs) must not become the k-fold training budget.
        out_cfg = set_training_epochs(out_cfg, model_key, args.train_epochs)

        config_parent = path.basename(path.dirname(args.config))
        if config_parent == "models":
            out_dir = "config_files/tuned/full"
        elif args.config.startswith("config_files/tuned/"):
            out_dir = path.dirname(args.config)
        else:
            out_dir = f"config_files/tuned/{config_parent}"
        os.makedirs(out_dir, exist_ok=True)
        out_name = path.basename(args.config)
        out_path = f"{out_dir}/{out_name}.yml"
        save_yaml(out_cfg, out_path)
        out_config_arg = f"{out_dir}/{out_name}"
        print(f"\nFinal config  → {out_path}")
        print(f"Full training → python run.py --config {out_config_arg} --gpu {gpus[0]}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sequential hyperparameter tuning with parallel GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Config path without .yml, e.g. config_files/tuned/emma_emotion")
    p.add_argument("--gpus", type=int, nargs="+",
                   help="GPU IDs to use, e.g. --gpus 0 1 2 3  (one candidate per GPU in parallel)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tune_epochs", type=int, default=15,
                   help="Max epochs per tuning trial (default: 15)")
    p.add_argument("--train_epochs", type=int, default=500,
                   help="Epoch budget written into the tuned config for the "
                        "final k-fold training runs (default: 500)")
    p.add_argument("--tune_fold", type=int, default=0,
                   help="Test fold index used for tuning (default: 0)")
    p.add_argument("--tune_order", nargs="+", default=None,
                   help="Order of params to tune. Defaults to config order.")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint JSON path for stop/resume support.")
    p.add_argument("--max_per_gpu", type=int, default=2,
                   help="Max candidates to run concurrently on each GPU "
                        "(default: 2). With more candidates than GPUs, extra "
                        "candidates are packed onto GPUs up to this limit "
                        "instead of waiting for a GPU to fully free up. If a "
                        "candidate OOMs while sharing a GPU, its retry "
                        "prefers running solo on a free GPU first. Set to 1 "
                        "to always run one candidate per GPU.")
    p.add_argument("--retry_oom", action="store_true",
                   help="Re-scan only candidates whose stored score is null "
                        "(crashed/OOM) for already-completed params, instead "
                        "of skipping completed params outright. Candidates "
                        "that already have a valid score are not re-run.")

    # Hidden worker-mode arguments (used when this script spawns itself)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker_gpu", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker_param", type=str, help=argparse.SUPPRESS)
    p.add_argument("--worker_value", type=str, help=argparse.SUPPRESS)
    p.add_argument("--worker_best_params", type=str, help=argparse.SUPPRESS)
    p.add_argument("--worker_model_key", type=str, help=argparse.SUPPRESS)
    p.add_argument("--worker_tmp_file", type=str, help=argparse.SUPPRESS)

    args = p.parse_args()

    if not args.worker and not args.gpus:
        p.error("--gpus is required, e.g. --gpus 0  or  --gpus 0 1 2 3")

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        main(args)

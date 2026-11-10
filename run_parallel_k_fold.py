"""Parallel k-fold runner with OOM-driven optimistic GPU scheduling.

Usage:
    python run_parallel_k_fold.py --config config_files/tuned/full/itemknn_onion \\
        --gpus 0 1 2 --folds 10
"""

import argparse
import datetime
import glob
import os
import signal
import subprocess
import sys
import time
from os import path as os_path

import yaml

import utils
from utils import is_oom_message, is_permanent_failure
from elliot.namespace.namespace_model_builder import NameSpaceBuilder

_HERE = os_path.abspath(os_path.dirname(__file__))

PYTHON_BIN = sys.executable


def resolve_run_tag(config_path: str) -> str:
    """Config tag run.py uses to name its intermediate fold files."""
    return config_path.replace("/", "_").replace("config_files_", "")


def resolve_merged_config(config_path: str) -> dict:
    """Merge config_files/DEFAULT.yml with the given config, as run.py does."""
    base_config = utils.load_yaml("config_files/DEFAULT.yml")
    extended_config = utils.load_yaml(config_path + ".yml")
    return utils.merge_dicts(base_config, extended_config)


def _resolve_output_path(config_path: str, attr: str) -> str:
    """Resolve a path_output_* namespace attribute as Elliot does."""
    config = resolve_merged_config(config_path)

    tmp_config_path = os.path.join("config_files", f"tmp_resolve_{os.getpid()}.yml")
    utils.save_yaml(config, tmp_config_path)
    try:
        builder = NameSpaceBuilder(tmp_config_path, _HERE, os_path.abspath(os_path.dirname(tmp_config_path)))
        return getattr(builder.base.base_namespace, attr)
    finally:
        os.remove(tmp_config_path)


def resolve_perf_dir(config_path: str) -> str:
    """Directory run.py writes the intermediate fold results to."""
    return _resolve_output_path(config_path, "path_output_rec_performance")


def resolve_recs_dir(config_path: str) -> str:
    """Directory models write their saved recommendation .tsv files to."""
    return _resolve_output_path(config_path, "path_output_rec_result")


def fold_done(perf_dir: str, run_tag: str, fold_idx: int) -> bool:
    """True if fold_idx's intermediate result has already been written."""
    fname = os.path.join(perf_dir, f"intermediate_fold_{run_tag}_{fold_idx:02d}.json")
    return os.path.exists(fname)


def build_run_cmd(config_path: str, seed: int, gpu: int, fold_idx: int,
                  warm_restart: bool = False) -> list:
    """Exactly one fold per subprocess: run.py --start_fold N --end_fold N+1."""
    cmd = [
        PYTHON_BIN, "-u", "run.py",
        "--config", config_path,
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--start_fold", str(fold_idx),
        "--end_fold", str(fold_idx + 1),
    ]
    if warm_restart:
        cmd.append("--warm_restart")
    return cmd


def attempt_log_path(log_dir: str, fold_idx: int, gpu: int, attempt: int) -> str:
    return os.path.join(log_dir, f"fold{fold_idx}_gpu{gpu}_attempt{attempt}.log")


def compute_pending(total_folds: int, perf_dir: str, run_tag: str) -> list:
    return [i for i in range(total_folds) if not fold_done(perf_dir, run_tag, i)]


def elliot_tmp_config_path(config_path: str, gpu: int, fold_idx: int) -> str:
    """Path of the tmp config run.py writes for a fold."""
    run_tag = resolve_run_tag(config_path)
    return f"config_files/tmp_{run_tag}_gpu{gpu}_fold{fold_idx}.yml"


def write_pids_file(pids_path: str, pids: list) -> None:
    with open(pids_path, "w") as f:
        for pid in pids:
            f.write(f"{pid}\n")


def kill_and_cleanup(running: dict, config_path: str, log_dir: str) -> None:
    """Kill all running fold subprocesses and remove their tmp configs and logs."""
    for fold_idx, gpu, attempt, proc, log_file, log_path in list(running.values()):
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass
        if os.path.exists(log_path):
            os.remove(log_path)
        tmp_cfg = elliot_tmp_config_path(config_path, gpu, fold_idx)
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)


def append_summary(log_dir: str, message: str) -> None:
    """Append a scheduling event to the run's summary log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with open(os.path.join(log_dir, "_summary.log"), "a") as f:
        f.write(line + "\n")


def schedule_folds(pending_folds, gpus, cmd_builder, fold_is_done_fn, log_dir,
                    max_per_gpu=None, config_path=None) -> tuple:
    """Run all pending folds across GPUs, reducing concurrency on CUDA OOM."""
    os.makedirs(log_dir, exist_ok=True)
    pids_path = os.path.join(log_dir, "pids.txt")
    results = {}
    tried_gpus = {idx: set() for idx in pending_folds}
    pending = list(pending_folds)
    running = {}
    next_slot_id = [0]
    global_cap = None

    def _gpu_load():
        load = {g: 0 for g in gpus}
        for _, gpu, _, _, _, _ in running.values():
            load[gpu] += 1
        return load

    def _effective_cap():
        caps = [c for c in (max_per_gpu, global_cap) if c is not None]
        return min(caps) if caps else None

    def _write_pids():
        write_pids_file(pids_path, [proc.pid for *_, proc, _, _ in running.values()])

    def _handle_interrupt(signum, frame):
        append_summary(log_dir,
            f"INTERRUPTED (signal {signum}) — killing {len(running)} running subprocess(es)")
        kill_and_cleanup(running, config_path, log_dir)
        write_pids_file(pids_path, [])
        sys.exit(1)

    def _launch(fold_idx, gpu):
        attempt = len(tried_gpus[fold_idx])
        slot_id = next_slot_id[0]
        next_slot_id[0] += 1
        log_path = attempt_log_path(log_dir, fold_idx, gpu, attempt)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd_builder(fold_idx, gpu),
                                 stdout=log_file, stderr=subprocess.STDOUT)
        tried_gpus[fold_idx].add(gpu)
        running[slot_id] = (fold_idx, gpu, attempt, proc, log_file, log_path)
        _write_pids()
        append_summary(log_dir, f"LAUNCH fold {fold_idx} on GPU {gpu} (attempt {attempt})")

    def _fill_idle_slots():
        while pending:
            load = _gpu_load()
            cap = _effective_cap()
            available = [g for g in gpus if cap is None or load[g] < cap]
            if not available:
                break
            available.sort(key=lambda g: load[g])
            launched = False
            for gpu in available:
                choice = next((idx for idx in pending if gpu not in tried_gpus[idx]), None)
                if choice is None:
                    continue
                pending.remove(choice)
                _launch(choice, gpu)
                launched = True
                break
            if not launched:
                break

    def _run_scheduling_loop():
        nonlocal global_cap
        _fill_idle_slots()

        while running:
            slot_id = None
            while slot_id is None:
                for sid, (_, _, _, proc, _, _) in running.items():
                    if proc.poll() is not None:
                        slot_id = sid
                        break
                if slot_id is None:
                    time.sleep(0.2)

            fold_idx, gpu, attempt, proc, log_file, log_path = running[slot_id]
            load_at_finish = _gpu_load()[gpu]  # includes this slot; not yet popped
            del running[slot_id]
            _write_pids()
            log_file.close()

            if fold_is_done_fn(fold_idx):
                results[fold_idx] = "DONE"
                append_summary(log_dir, f"DONE fold {fold_idx} on GPU {gpu} (attempt {attempt})")
            else:
                log_text = ""
                try:
                    with open(log_path) as f:
                        log_text = f.read()
                except OSError:
                    pass

                if is_oom_message(log_text):
                    if global_cap is None:
                        global_cap = max(1, load_at_finish - 1)
                    else:
                        global_cap = max(1, global_cap - 1)

                    if len(tried_gpus[fold_idx]) < len(gpus):
                        pending.append(fold_idx)
                        append_summary(log_dir,
                            f"OOM fold {fold_idx} on GPU {gpu} (attempt {attempt}) "
                            f"-> global_cap={global_cap}, requeued")
                    else:
                        results[fold_idx] = "FAILED"
                        append_summary(log_dir,
                            f"FAILED fold {fold_idx} (OOM'd on every GPU {gpus}) "
                            f"-> global_cap={global_cap}")
                else:
                    results[fold_idx] = "FAILED"
                    reason = "permanent error" if is_permanent_failure(log_text) else "crash"
                    append_summary(log_dir,
                        f"FAILED fold {fold_idx} on GPU {gpu} (attempt {attempt}, "
                        f"{reason}, exit {proc.returncode})")

            if os.path.exists(log_path):
                os.remove(log_path)

            _fill_idle_slots()

    old_handlers = {
        sig: signal.signal(sig, _handle_interrupt)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        _run_scheduling_loop()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    return results, global_cap


def sweep_stray_attempt_logs(log_dir: str) -> list:
    """Remove leftover per-attempt capture logs."""
    if not os.path.isdir(log_dir):
        return []
    removed = []
    for fname in os.listdir(log_dir):
        if "attempt" in fname and fname.endswith(".log"):
            path = os.path.join(log_dir, fname)
            os.remove(path)
            removed.append(path)
    return removed


def find_fold_recs(recs_dir: str, model_prefix: str, fold_idx: int) -> str:
    """Locate the saved best-epoch recs .tsv for one model/fold."""
    matches = glob.glob(os.path.join(recs_dir, f"{model_prefix}_*_fold={fold_idx}.tsv"))
    matches += glob.glob(os.path.join(recs_dir, f"{model_prefix}_fold={fold_idx}.tsv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


_RERANKER_RECS_PREFIX = {
    "RerankingRecommender": "Reranker",
    "RerankingRecommender2": "Reranker",
    "RandomShuffleReranker": "RandomShuffle",
    "CalibratedReranker": "CalibratedReranker",
}


def reranker_recs_prefixes(config: dict) -> list:
    """Extra recs-filename prefixes for reranking models."""
    prefixes = []
    for key, params in config["experiment"]["models"].items():
        cls = key.split(".", 1)[-1]
        stem = _RERANKER_RECS_PREFIX.get(cls)
        if stem is None or not isinstance(params, dict):
            continue
        input_path = params.get("input_path")
        if not input_path:
            continue
        backbone = os.path.basename(str(input_path)).split("_")[0]
        prefix = f"{stem}{backbone}"
        metric = params.get("calibration_metric") if stem == "CalibratedReranker" else None
        if metric:
            prefix = f"{prefix}_*_{metric}"
        prefixes.append(prefix)
    return prefixes


def eval_done(perf_dir: str, run_tag: str, fold_idx: int) -> bool:
    """True if this fold's offline evaluation JSON already exists."""
    fname = os.path.join(perf_dir, f"intermediate_fold_eval_{run_tag}_{fold_idx:02d}.json")
    return os.path.exists(fname)


def build_eval_cmd(config_path: str, fold_idx: int, recs_path: str) -> list:
    return [
        PYTHON_BIN, "-u", "run_eval.py",
        "--tuned_config", config_path,
        "--fold", str(fold_idx),
        "--path", recs_path,
    ]


def run_offline_evals(config_path: str, perf_dir: str, run_tag: str, total_folds: int) -> dict:
    """Evaluate each fold's saved recommendations with run_eval.py."""
    config = resolve_merged_config(config_path)
    model_prefixes = [key.split(".", 1)[-1] for key in config["experiment"]["models"].keys()]
    model_prefixes += reranker_recs_prefixes(config)
    recs_dir = resolve_recs_dir(config_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/offline_eval_{timestamp}"

    results = {}
    running = {}  # fold_idx -> (proc, log_file, log_path)
    for fold_idx in range(total_folds):
        if eval_done(perf_dir, run_tag, fold_idx):
            results[fold_idx] = "ALREADY_DONE"
            continue
        recs_path = None
        for prefix in model_prefixes:
            recs_path = find_fold_recs(recs_dir, prefix, fold_idx)
            if recs_path:
                break
        if recs_path is None:
            results[fold_idx] = "SKIPPED"
            print(f"[offline eval] fold {fold_idx}: no recs file matching "
                  f"{model_prefixes} *_fold={fold_idx}.tsv in {recs_dir} — skipped")
            continue
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"eval_fold{fold_idx}.log")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(build_eval_cmd(config_path, fold_idx, recs_path),
                                stdout=log_file, stderr=subprocess.STDOUT)
        running[fold_idx] = (proc, log_file, log_path)
        print(f"[offline eval] fold {fold_idx}: evaluating {os.path.basename(recs_path)}")

    for fold_idx, (proc, log_file, log_path) in running.items():
        proc.wait()
        log_file.close()
        if proc.returncode == 0 and eval_done(perf_dir, run_tag, fold_idx):
            results[fold_idx] = "DONE"
            os.remove(log_path)
        else:
            results[fold_idx] = "FAILED"
            print(f"[offline eval] fold {fold_idx}: FAILED (exit {proc.returncode}), "
                  f"log kept at {log_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Run k-fold cross-validation in parallel, with OOM-driven "
                     "optimistic GPU scheduling.")
    p.add_argument("--config", required=True,
                   help="Config path without .yml extension")
    p.add_argument("--gpus", type=int, nargs="+", required=True,
                   help="GPU indices to schedule folds across")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--folds", type=int, default=10,
                   help="Total number of folds")
    p.add_argument("--max_per_gpu", type=int, default=None,
                   help="Optional ceiling on initial concurrent folds per GPU "
                        "(default: unbounded — every pending fold launches "
                        "immediately). The reactive OOM-driven global cap can "
                        "only push concurrency below this ceiling, never above it.")
    p.add_argument("--no_offline_eval", action="store_true",
                   help="Skip the automatic post-training offline full-metric "
                        "evaluation of each fold's saved best-epoch recs "
                        "(run_eval.py can still be invoked manually).")
    p.add_argument("--warm_restart", action="store_true",
                   help="Resume interrupted folds from their last validation-epoch "
                        "resume checkpoint instead of epoch 0 (preserves "
                        "early-stopping history; requires save_weights-capable "
                        "models — LightGCN/DeepMF/SiBraR/LightGCNM).")
    p.add_argument("--pci_order", action="store_true",
                   help="Set CUDA_DEVICE_ORDER=PCI_BUS_ID so GPU indices match "
                        "nvidia-smi/PCI bus order.")
    return p.parse_args()


def main(args):
    print(f"Starting run_parallel_k_fold.py for config: {args.config}")
    print(yaml.dump(resolve_merged_config(args.config), default_flow_style=False, sort_keys=False))

    if args.pci_order:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print(f"GPUs: {' '.join(str(g) for g in args.gpus)}")
    if args.pci_order:
        print("CUDA_DEVICE_ORDER: PCI_BUS_ID")

    run_tag = resolve_run_tag(args.config)
    perf_dir = resolve_perf_dir(args.config)
    pending = compute_pending(args.folds, perf_dir, run_tag)

    print(f"Config    : {args.config}")
    print(f"Run tag   : {run_tag}")
    print(f"Perf dir  : {perf_dir}")
    print(f"Pending   : {len(pending)} of {args.folds} fold(s) "
          f"({args.folds - len(pending)} already done)")
    print("")

    if not pending:
        print("All folds already done — nothing to run.")
        if not args.no_offline_eval:
            _report_offline_evals(run_offline_evals(args.config, perf_dir, run_tag, args.folds))
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/parallel_{timestamp}"
    print(f"Logs      : {log_dir}")
    print("")

    def cmd_builder(fold_idx, gpu):
        return build_run_cmd(args.config, args.seed, gpu, fold_idx,
                             warm_restart=args.warm_restart)

    def fold_is_done_fn(fold_idx):
        return fold_done(perf_dir, run_tag, fold_idx)

    results, final_cap = schedule_folds(pending, args.gpus, cmd_builder, fold_is_done_fn,
                                         log_dir, max_per_gpu=args.max_per_gpu,
                                         config_path=args.config)

    stray = sweep_stray_attempt_logs(log_dir)
    if stray:
        append_summary(log_dir, f"Removed {len(stray)} stray attempt-log file(s) missed by normal cleanup")

    failed = sorted(idx for idx, status in results.items() if status == "FAILED")
    append_summary(log_dir,
        f"RUN COMPLETE: {len(results) - len(failed)} done, {len(failed)} failed")

    print("")
    if final_cap is not None:
        print(f"Global concurrency cap reduced to {final_cap} due to OOM.")
    print("Results:")
    for fold_idx in sorted(results):
        print(f"  fold {fold_idx:2d}: {results[fold_idx]}")

    # Offline full-metric evaluation runs even when some folds failed —
    # completed folds' recs are already on disk and worth evaluating.
    if not args.no_offline_eval:
        _report_offline_evals(run_offline_evals(args.config, perf_dir, run_tag, args.folds))

    if failed:
        print(f"\n{len(failed)} fold(s) failed: {failed} — re-run to retry them.")


def _report_offline_evals(eval_results: dict) -> None:
    print("\nOffline full-metric eval:")
    for fold_idx in sorted(eval_results):
        print(f"  fold {fold_idx:2d}: {eval_results[fold_idx]}")


if __name__ == "__main__":
    main(parse_args())

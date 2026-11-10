"""Build `onion186`: an Onion subsample matched to EMMA's post-k-core shape.

Usage:
    python build_onion186.py --dry-run          # report the shape, write nothing
    python build_onion186.py                    # write the dataset
"""

import argparse
import collections
import hashlib
import os
import shutil

import numpy as np

DATA_ROOT = "/opt/datasets/Music4All/music4all/onion_elliot"
SOURCE = "onion"
TARGET = "onion186"
REFERENCE = "emma"

# EMMA's post-k-core shape -- the thing being matched. Recomputed and asserted
# by `emma_target_shape()` rather than trusted as a constant.
EMMA_SHAPE = (3776, 186, 28112)

K_CORE = 5
N_ITEMS = 186
N_USERS = 3776
PER_USER_CAP = 10
SEED = 0

EMBEDDING_DIRS = ("audio_embeddings", "textual_embeddings", "visual_embeddings",
                  "emotion_embeddings")


def read_interactions(dataset, data_root=DATA_ROOT):
    """Read a dataset's 4-column TSV as (user, item, rating, timestamp) tuples."""
    path = os.path.join(data_root, dataset, "interactions.tsv")
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return rows


def iterative_k_core(rows, core=K_CORE):
    """Elliot's `iterative_k_core` prefiltering, reimplemented for measurement."""
    current = rows
    while True:
        user_counts = collections.Counter(r[0] for r in current)
        item_counts = collections.Counter(r[1] for r in current)
        nxt = [r for r in current
               if user_counts[r[0]] >= core and item_counts[r[1]] >= core]
        if len(nxt) == len(current):
            return current
        current = nxt


def shape(rows):
    """(n_users, n_items, n_interactions)."""
    return (len({r[0] for r in rows}), len({r[1] for r in rows}), len(rows))


def emma_target_shape(data_root=DATA_ROOT):
    """EMMA's post-k-core shape, measured rather than assumed."""
    return shape(iterative_k_core(read_interactions(REFERENCE, data_root)))


def _embedding_hash(path):
    array = np.ascontiguousarray(np.load(path), dtype=np.float64)
    return hashlib.sha1(array.tobytes()).hexdigest()


def emma_track_ids_in_source(data_root=DATA_ROOT, modality="audio_embeddings",
                            variant="maest"):
    """Onion item ids whose audio embedding is byte-identical to an EMMA track."""
    source_dir = os.path.join(data_root, SOURCE, modality, variant)
    reference_dir = os.path.join(data_root, REFERENCE, modality, variant)

    reference_hashes = {
        _embedding_hash(os.path.join(reference_dir, f))
        for f in os.listdir(reference_dir) if f.endswith(".npy")
    }
    excluded = set()
    for f in os.listdir(source_dir):
        if not f.endswith(".npy"):
            continue
        if _embedding_hash(os.path.join(source_dir, f)) in reference_hashes:
            excluded.add(int(f[:-4]))
    return excluded


def subsample(rows, excluded_items, n_items=N_ITEMS, n_users=N_USERS,
              cap=PER_USER_CAP, core=K_CORE, seed=SEED):
    """Select the subsample. Returns rows *before* the final k-core."""
    rng = np.random.default_rng(seed)

    eligible = [r for r in rows if r[1] not in excluded_items]
    item_counts = collections.Counter(r[1] for r in eligible)
    pool = {item for item, _ in item_counts.most_common(n_items)}
    in_pool = [r for r in eligible if r[1] in pool]

    by_user = collections.defaultdict(list)
    for row in in_pool:
        by_user[row[0]].append(row)

    # Only users who would survive the core anyway -- sampling users who are
    # about to be filtered out would silently undershoot the user target.
    candidates = sorted(u for u, v in by_user.items() if len(v) >= core)
    if n_users < len(candidates):
        chosen = set(rng.choice(candidates, size=n_users, replace=False).tolist())
    else:
        chosen = set(candidates)

    out = []
    for user in sorted(chosen):
        interactions = by_user[user]
        if len(interactions) > cap:
            idx = rng.choice(len(interactions), size=cap, replace=False)
            out.extend(interactions[j] for j in sorted(idx))
        else:
            out.extend(interactions)
    return out


def build(data_root=DATA_ROOT, seed=SEED, excluded_items=None):
    """Full pipeline. Returns (final_rows, excluded_items)."""
    rows = read_interactions(SOURCE, data_root)
    if excluded_items is None:
        excluded_items = emma_track_ids_in_source(data_root)
    return iterative_k_core(subsample(rows, excluded_items, seed=seed)), excluded_items


def write_dataset(rows, data_root=DATA_ROOT, target=TARGET, force=False):
    """Write interactions.tsv and symlink the embedding trees."""
    target_dir = os.path.join(data_root, target)
    interactions = os.path.join(target_dir, "interactions.tsv")
    if os.path.exists(interactions) and not force:
        raise FileExistsError(
            f"{interactions} already exists; pass force=True to overwrite. "
            "Silently rebuilding would invalidate any splits already derived "
            "from the previous version."
        )
    os.makedirs(target_dir, exist_ok=True)

    with open(interactions, "w") as fh:
        for user, item, rating, timestamp in rows:
            fh.write(f"{user}\t{item}\t{rating}\t{timestamp}\n")

    for name in EMBEDDING_DIRS:
        source = os.path.join(data_root, SOURCE, name)
        link = os.path.join(target_dir, name)
        if not os.path.exists(source):
            continue
        if os.path.islink(link) or os.path.exists(link):
            if force:
                (os.unlink if os.path.islink(link) else shutil.rmtree)(link)
            else:
                continue
        os.symlink(source, link)
    return target_dir


def report(rows, target_shape, excluded_items):
    users, items, interactions = shape(rows)
    t_users, t_items, t_inter = target_shape
    lines = [
        "onion186 — Onion subsampled to EMMA's post-k-core shape",
        "=" * 66,
        "",
        f"EMMA tracks excluded from the item pool: {len(excluded_items)} "
        f"(matched by audio-embedding content hash, not item id)",
        "",
        f"{'':<16}{'onion186':>12}{'EMMA':>12}{'error':>12}",
        "-" * 52,
    ]
    for label, got, want in (("users", users, t_users), ("items", items, t_items),
                             ("interactions", interactions, t_inter)):
        err = "exact" if got == want else f"{(got - want) / want * 100:+.2f}%"
        lines.append(f"{label:<16}{got:>12,}{want:>12,}{err:>12}")

    density = interactions / (users * items) if users and items else float("nan")
    t_density = t_inter / (t_users * t_items)
    lines += [
        "",
        f"{'density':<16}{density:>12.4f}{t_density:>12.4f}"
        f"{(density - t_density) / t_density * 100:>+11.2f}%",
        f"{'inter/user':<16}{interactions / users:>12.1f}{t_inter / t_users:>12.1f}",
        f"{'inter/item':<16}{interactions / items:>12.1f}{t_inter / t_items:>12.1f}",
        "",
        "All counts are measured after re-applying iterative_k_core: 5 — the",
        "shape the models actually see.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the resulting shape without writing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing onion186 dataset")
    args = parser.parse_args()

    rows, excluded = build(args.data_root, seed=args.seed)
    print(report(rows, emma_target_shape(args.data_root), excluded))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    target_dir = write_dataset(rows, args.data_root, force=args.force)
    print(f"\nWrote {target_dir}")
    print("Next: python precompute_splits.py --datasets onion186")


if __name__ == "__main__":
    main()

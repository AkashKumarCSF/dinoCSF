import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

DATA_ROOT = Path("/home/administrator/Akash/datasets/CSF/")
ODS_FILE = Path("data/supplementary_table_73.ods")

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

TOLERANCE = 0.05   # allow ±5% drift


# ==========================================================
# STEP 1: BUILD CASE → CLASS COUNTS
# ==========================================================

def build_case_class_counts(data_root, ods_file):

    df = pd.read_excel(ods_file, engine="odf")

    df["Case"] = df["Case"].ffill()
    df = df.dropna(subset=["Case", "Index"])

    df["Case"] = df["Case"].astype(int) + 1
    df["Index"] = df["Index"].astype(str).str.strip()

    index_to_case = {
        row["Index"]: int(row["Case"])
        for _, row in df.iterrows()
    }

    liquor_pattern = re.compile(r"Liquor_\d+")

    case_class_counts = defaultdict(lambda: defaultdict(int))
    samples = []

    for class_dir in data_root.iterdir():

        if not class_dir.is_dir():
            continue

        cls = class_dir.name

        for img in class_dir.rglob("*.png"):

            m = liquor_pattern.search(img.stem)
            if not m:
                continue

            liquor_id = m.group(0)

            if liquor_id not in index_to_case:
                continue

            case = index_to_case[liquor_id]

            case_class_counts[case][cls] += 1

            samples.append((str(img), cls, case, liquor_id))

    return case_class_counts, samples


# ==========================================================
# STEP 2: GLOBAL STATS
# ==========================================================

def compute_global(case_class_counts):

    global_counts = defaultdict(int)

    for cdata in case_class_counts.values():
        for k, v in cdata.items():
            global_counts[k] += v

    total = sum(global_counts.values())

    return global_counts, total


# ==========================================================
# STEP 3: SAFE CONSTRAINED SPLIT
# ==========================================================

def split_cases(case_class_counts):

    classes = sorted({
        c for d in case_class_counts.values() for c in d
    })

    global_counts, total = compute_global(case_class_counts)

    target_size = {
        "train": TRAIN_RATIO * total,
        "val":   VAL_RATIO * total,
        "test":  TEST_RATIO * total,
    }

    current_size = defaultdict(int)

    current_class = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    assignment = {}

    case_order = sorted(
        case_class_counts.keys(),
        key=lambda c: sum(case_class_counts[c].values()),
        reverse=True
    )

    # --------------------------------------------------
    # HARD constraint: capacity check
    # --------------------------------------------------
    def can_fit(case_id, split):

        case_size = sum(case_class_counts[case_id].values())

        return current_size[split] + case_size <= target_size[split] * (1 + TOLERANCE)

    # --------------------------------------------------
    # scoring function (soft)
    # --------------------------------------------------
    def score(case_id, split):

        penalty = 0.0

        # class imbalance penalty
        for cls in classes:

            after = current_class[split][cls] + case_class_counts[case_id].get(cls, 0)

            diff = after - (global_counts[cls] * {
                "train": TRAIN_RATIO,
                "val": VAL_RATIO,
                "test": TEST_RATIO
            }[split])

            penalty += diff * diff

        # mild size penalty
        size_after = current_size[split] + sum(case_class_counts[case_id].values())
        penalty += 0.001 * (size_after - target_size[split]) ** 2

        return penalty

    # --------------------------------------------------
    # assignment loop
    # --------------------------------------------------
    for case_id in case_order:

        candidates = []

        for split in ["train", "val", "test"]:

            if can_fit(case_id, split):
                candidates.append(split)

        # fallback if all full → relax constraint
        if not candidates:
            candidates = ["train", "val", "test"]

        best_split = min(
            candidates,
            key=lambda s: score(case_id, s)
        )

        assignment[case_id] = best_split

        size = sum(case_class_counts[case_id].values())
        current_size[best_split] += size

        for cls, v in case_class_counts[case_id].items():
            current_class[best_split][cls] += v

    return assignment


# ==========================================================
# STEP 4: BUILD FINAL SPLITS
# ==========================================================

def build_samples(samples, assignment):

    train, val, test = [], [], []

    for s in samples:

        split = assignment[s[2]]

        if split == "train":
            train.append(s)
        elif split == "val":
            val.append(s)
        else:
            test.append(s)

    return train, val, test


# ==========================================================
# STEP 5: PRINT STATS
# ==========================================================

def print_stats(samples, name):

    c = Counter(s[1] for s in samples)
    total = sum(c.values())

    print("\n" + "="*60)
    print(name)
    print("="*60)

    for k in sorted(c):
        print(f"{k:<30} {c[k]:>8} ({100*c[k]/total:6.2f}%)")

    print("-"*60)
    print("TOTAL:", total)


# ==========================================================
# MAIN
# ==========================================================

def main():

    case_class_counts, samples = build_case_class_counts(
        DATA_ROOT,
        ODS_FILE
    )

    assignment = split_cases(case_class_counts)

    train, val, test = build_samples(samples, assignment)

    print_stats(train, "TRAIN")
    print_stats(val, "VAL")
    print_stats(test, "TEST")


if __name__ == "__main__":
    main()
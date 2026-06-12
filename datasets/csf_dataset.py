import re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Dataset
# ============================================================
class CSFLiquorDataset(Dataset):
    def __init__(self, samples, transform=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(set(s[1] for s in samples))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name, case_id, liquor_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[class_name]
        return img, label


# ============================================================
# Build case-level counts
# ============================================================
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

    for class_dir in Path(data_root).iterdir():

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


# ============================================================
# ORIGINAL SPLITTER (UNCHANGED LOGIC)
# ============================================================
def split_cases(case_class_counts):

    classes = sorted({
        c for d in case_class_counts.values() for c in d
    })

    global_counts = defaultdict(int)

    for d in case_class_counts.values():
        for c, v in d.items():
            global_counts[c] += v

    total = sum(global_counts.values())

    target_ratio = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    }

    target_size = {
        k: v * total for k, v in target_ratio.items()
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

    def can_fit(case_id, split):

        case_size = sum(case_class_counts[case_id].values())

        return current_size[split] + case_size <= target_size[split] * 1.05

    def score(case_id, split):

        penalty = 0.0

        for cls in classes:

            after = current_class[split][cls] + case_class_counts[case_id].get(cls, 0)

            target = global_counts[cls] * target_ratio[split]

            penalty += (after - target) ** 2

        size_after = current_size[split] + sum(case_class_counts[case_id].values())
        penalty += 0.001 * (size_after - target_size[split]) ** 2

        return penalty

    for case_id in case_order:

        candidates = []

        for split in ["train", "val", "test"]:
            if can_fit(case_id, split):
                candidates.append(split)

        if not candidates:
            candidates = ["train", "val", "test"]

        best_split = min(candidates, key=lambda s: score(case_id, s))

        assignment[case_id] = best_split

        size = sum(case_class_counts[case_id].values())
        current_size[best_split] += size

        for cls, v in case_class_counts[case_id].items():
            current_class[best_split][cls] += v

    return assignment


def print_class_distribution(samples, split_name):
    counts = Counter([s[1] for s in samples])  # s[1] = class_name

    print(f"\n{'='*60}")
    print(f"{split_name} CLASS DISTRIBUTION")
    print(f"{'='*60}")

    total = sum(counts.values())

    for cls in sorted(counts.keys()):
        n = counts[cls]
        pct = 100 * n / total
        print(f"{cls:<25} {n:>8} ({pct:6.2f}%)")

    print("-"*60)
    print(f"TOTAL{'':<20} {total:>8}")


# ============================================================
# BUILD DATASETS
# ============================================================
def build_csf_datasets(data_root, ods_file, transform=None):

    case_class_counts, samples = build_case_class_counts(
        data_root,
        ods_file
    )

    assignment = split_cases(case_class_counts)

    train_samples, val_samples, test_samples = [], [], []

    for s in samples:

        split = assignment[s[2]]

        if split == "train":
            train_samples.append(s)
        elif split == "val":
            val_samples.append(s)
        else:
            test_samples.append(s)

    all_classes = sorted(set(s[1] for s in samples))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    print_class_distribution(train_samples, "TRAIN")
    print_class_distribution(val_samples, "VAL")
    print_class_distribution(test_samples, "TEST")

    train_ds = CSFLiquorDataset(train_samples, transform, class_to_idx)
    val_ds   = CSFLiquorDataset(val_samples, transform, class_to_idx)
    test_ds  = CSFLiquorDataset(test_samples, transform, class_to_idx)

    train_counts = Counter(s[1] for s in train_samples)
    return train_ds, val_ds, test_ds, class_to_idx, train_counts
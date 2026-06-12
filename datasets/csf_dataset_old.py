import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Dataset Class
# ============================================================
class CSFLiquorDataset(Dataset):
    def __init__(self, samples, transform=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform

        # IMPORTANT: shared mapping (ensures consistency across splits)
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
# Core builder
# ============================================================
def get_class_counts(samples):
    """
    Returns raw class counts from samples.
    Fast, no image loading.
    """
    return Counter([s[1] for s in samples])


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


def build_csf_datasets(data_root, split_json_path, ods_file_path, transform=None):

    # -------------------------
    # LOAD SPLITS
    # -------------------------
    with open(split_json_path, "r") as f:
        split = json.load(f)

    train_cases = set(split["train"])
    val_cases = set(split["val"])
    test_cases = set(split["test"])

    # -------------------------
    # LOAD ODS
    # -------------------------
    df = pd.read_excel(ods_file_path, engine="odf")

    df["Case"] = df["Case"].ffill()
    df = df.dropna(subset=["Case", "Index"])

    df["Case"] = df["Case"].astype(int) + 1
    df["Index"] = df["Index"].astype(str).str.strip()

    # -------------------------
    # mappings
    # -------------------------
    case_to_indices = defaultdict(set)

    for _, row in df.iterrows():
        case_to_indices[int(row["Case"])].add(row["Index"])

    index_to_case = {}
    for case_id, liqs in case_to_indices.items():
        for l in liqs:
            index_to_case[l] = case_id

    # -------------------------
    # scan dataset
    # -------------------------
    liquor_pattern = re.compile(r"Liquor_\d+")

    dataset_root = Path(data_root)

    train_samples, val_samples, test_samples = [], [], []

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for img_path in class_dir.rglob("*.png"):

            match = liquor_pattern.search(img_path.stem)
            if match is None:
                continue

            liquor_id = match.group(0)

            if liquor_id not in index_to_case:
                continue

            case_id = index_to_case[liquor_id]

            sample = (str(img_path), class_name, case_id, liquor_id)

            if case_id in train_cases:
                train_samples.append(sample)
            elif case_id in val_cases:
                val_samples.append(sample)
            elif case_id in test_cases:
                test_samples.append(sample)

    print_class_distribution(train_samples, "TRAIN")
    print_class_distribution(val_samples, "VAL")
    print_class_distribution(test_samples, "TEST")
    #exit()
    # -------------------------
    # build shared class mapping
    # -------------------------
    all_classes = sorted(
        set(s[1] for s in train_samples + val_samples + test_samples)
    )
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # -------------------------
    # datasets
    # -------------------------
    train_ds = CSFLiquorDataset(train_samples, transform, class_to_idx)
    val_ds   = CSFLiquorDataset(val_samples, transform, class_to_idx)
    test_ds  = CSFLiquorDataset(test_samples, transform, class_to_idx)
    #print("computing train count")
    train_counts = Counter([s[1] for s in train_samples])
    num_classes = len(class_to_idx)

    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    #print("computing class count")
    for cls_name, idx in class_to_idx.items():
        class_counts[idx] = train_counts[cls_name]
    #print("completed")
    exit()
    return train_ds, val_ds, test_ds, class_to_idx, class_counts
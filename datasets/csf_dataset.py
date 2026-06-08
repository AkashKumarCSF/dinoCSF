import json
import re
from pathlib import Path
from collections import defaultdict
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

    return train_ds, val_ds, test_ds
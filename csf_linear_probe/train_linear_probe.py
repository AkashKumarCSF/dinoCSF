import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:

    experiment_id = 1
    dataset_name: str = "CSF"
    root_dir: str = "/home/administrator/Akash/datasets/CSF/"
    split_json: str = "/home/administrator/Akash/datasets/split_file_15cls.json"
    project_dir: str = "/home/administrator/Akash/pycharm_projects/Selfsupervised/dinoCSF/"

    img_size: int = 224
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 100
    num_classes: int = 15

    num_workers: int = 4


# =========================================================
# DDP HELPERS
# =========================================================

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(cfg):
    logging.basicConfig(
        filename=cfg.project_dir + "logs/Experiment" + str(cfg.experiment_id) + "_training.log",
        filemode="a",
        format="%(asctime)s | %(message)s",
        level=logging.INFO
    )
    return logging.getLogger()


def setup_tensorboard():
    return SummaryWriter(log_dir="runs/")


def compute_confusion_matrix(outputs, labels, num_classes):
    preds = outputs.argmax(1)

    cm = torch.zeros((num_classes, num_classes), device=outputs.device)

    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1

    return cm

def compute_metrics(cm: torch.Tensor):

    cm = cm.cpu().numpy()

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) != 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = sensitivity

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0
    )

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": np.mean(f1)
    }


# =========================================================
# CSF DATASET (PID SPLIT ONLY FOR CSF)
# =========================================================

class CSFSplitDataset(Dataset):
    """
    CSF dataset with PID-based filtering using filename.
    """

    def __init__(self, root: str, split_pids: set, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        base = datasets.ImageFolder(root)

        for path, label in base.samples:
            filename = os.path.basename(path)

            try:
                pid = int(filename.split("_")[1])
            except Exception:
                continue

            if pid in split_pids:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# DATASET FACTORY
# =========================================================

def load_split_json(path: str):
    with open(path, "r") as f:
        split = json.load(f)
    return {
        "train": set(split["train"]),
        "val": set(split["val"]),
        "test": set(split["test"]),
    }


def build_datasets(cfg: Config, transform):
    """
    Only CSF uses JSON PID splitting.
    Other datasets fall back to ImageFolder default behavior.
    """

    if cfg.dataset_name.upper() == "CSF":
        splits = load_split_json(cfg.split_json)

        train_ds = CSFSplitDataset(cfg.root_dir, splits["train"], transform)
        val_ds = CSFSplitDataset(cfg.root_dir, splits["val"], transform)
        test_ds = CSFSplitDataset(cfg.root_dir, splits["test"], transform)

        return train_ds, val_ds, test_ds

    else:
        # Generic fallback (no PID split)
        base = datasets.ImageFolder(cfg.root_dir, transform=transform)

        n = len(base)
        train_len = int(0.8 * n)
        val_len = int(0.1 * n)

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            base,
            [train_len, val_len, n - train_len - val_len]
        )

        return train_ds, val_ds, test_ds


# =========================================================
# DATALOADERS
# =========================================================

def build_loaders(cfg: Config, train_ds, val_ds, test_ds):

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_sampler


# =========================================================
# MODEL SETUP
# =========================================================

def build_model(cfg: Config, device):
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14"
    ).to(device)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    classifier = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 15)
    ).to(device)

    classifier = nn.parallel.DistributedDataParallel(
        classifier,
        device_ids=[int(os.environ.get("LOCAL_RANK", 0))]
    )

    return backbone, classifier


# =========================================================
# TRAIN / EVAL UTILITIES
# =========================================================

def extract_features(backbone, x):
    with torch.no_grad():
        feats = backbone.forward_features(x)
        return feats["x_norm_clstoken"]


def train_one_epoch(backbone, classifier, loader, sampler, optimizer, criterion, device, epoch, rank):

    classifier.train()
    sampler.set_epoch(epoch)

    total_loss = 0.0

    for imgs, labels in tqdm(loader, disable=(rank != 0)):

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = extract_features(backbone, imgs)
        outputs = classifier(feats)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(backbone, classifier, loader, criterion, device, num_classes):

    classifier.eval()

    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)

    cm = torch.zeros((num_classes, num_classes), device=device)

    for imgs, labels in loader:

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = extract_features(backbone, imgs)
        outputs = classifier(feats)

        loss = criterion(outputs, labels)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum()
        total += labels.size(0)
        loss_sum += loss.detach()

        # update confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    dist.all_reduce(correct)
    dist.all_reduce(total)
    dist.all_reduce(loss_sum)
    dist.all_reduce(cm)

    acc = correct.item() / total.item()
    avg_loss = loss_sum.item() / dist.get_world_size()

    metrics = compute_metrics(cm)

    return acc, avg_loss, cm, metrics


# =========================================================
# MAIN
# =========================================================

def main():

    cfg = Config()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank()
    logger = setup_logging(cfg)

    writer = setup_tensorboard() if rank == 0 else None

    # -------------------------
    # Transforms
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    # -------------------------
    # Dataset
    # -------------------------
    train_ds, val_ds, test_ds = build_datasets(cfg, transform)

    # -------------------------
    # Loader
    # -------------------------
    train_loader, val_loader, test_loader, train_sampler = build_loaders(
        cfg, train_ds, val_ds, test_ds
    )

    # -------------------------
    # Model
    # -------------------------
    backbone, classifier = build_model(cfg, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg.lr)

    best_acc = 0.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(cfg.epochs):

        train_loss = train_one_epoch(
            backbone, classifier,
            train_loader,
            train_sampler,
            optimizer,
            criterion,
            device,
            epoch,
            rank
        )

        val_acc, val_loss, cm, metrics = evaluate(
            backbone,
            classifier,
            val_loader,
            criterion,
            device,
            cfg.num_classes
        )

        if rank == 0:

            log_msg = (
                f"Epoch {epoch} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"ValAcc={val_acc:.4f} | "
                f"MacroF1={metrics['macro_f1']:.4f}"
            )

            print("\n" + log_msg)
            logger.info(log_msg)

            # TensorBoard logs
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/macro", metrics["macro_f1"], epoch)

            # per-class metrics
            for i in range(cfg.num_classes):
                writer.add_scalar(f"F1/class_{i}", metrics["f1"][i], epoch)
                writer.add_scalar(f"Sensitivity/class_{i}", metrics["sensitivity"][i], epoch)
                writer.add_scalar(f"Specificity/class_{i}", metrics["specificity"][i], epoch)

            # save best model
            check_point_path = os.path.join(cfg.project_dir, "checkpoints", f"Experiment_{cfg.experiment_id}")
            os.makedirs(check_point_path, exist_ok=True)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(classifier.module.state_dict(), check_point_path + "/Epoch_" + str(epoch) + "_best_linear_probe.pth")
                print("Saved best model")

    # -------------------------
    # Test
    # -------------------------
    test_acc, test_loss, test_cm, test_metrics = evaluate(
        backbone,
        classifier,
        test_loader,
        criterion,
        device,
        cfg.num_classes
    )

    if rank == 0:
        print("\nFINAL TEST RESULTS")
        print(f"ACC: {test_acc:.4f}")
        print(f"Macro F1: {test_metrics['macro_f1']:.4f}")

        logger.info(f"TEST_ACC={test_acc:.4f} TEST_F1={test_metrics['macro_f1']:.4f}")

        writer.add_scalar("Test/accuracy", test_acc)
        writer.add_scalar("Test/macro_f1", test_metrics["macro_f1"])

    cleanup_ddp()


if __name__ == "__main__":
    main()
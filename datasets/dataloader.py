from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets.csf_dataset import build_csf_datasets


def build_dataloaders(cfg, transform):

    train_ds, val_ds, test_ds, class_to_idx, class_counts  = build_csf_datasets(
        data_root=cfg["dataset"]["data_root"],
        ods_file=cfg["dataset"]["ods_file"],
        transform=transform
    )

    train_sampler = DistributedSampler(
        train_ds,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_ds,
        shuffle=False
    )

    test_sampler = DistributedSampler(
        test_ds,
        shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=test_sampler,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        class_to_idx,
        class_counts
    )
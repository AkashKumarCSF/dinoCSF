import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


# -------------------------
# DDP setup
# -------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# -------------------------
# Config
# -------------------------
BATCH_SIZE = 32  # per GPU
LR = 1e-3
EPOCHS = 10
NUM_CLASSES = 15
IMG_SIZE = 224


# -------------------------
# Train
# -------------------------
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # -------------------------
    # Frozen DINOv2 backbone
    # -------------------------
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14"
    ).to(device)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False


    # -------------------------
    # Classifier head
    # -------------------------
    classifier = nn.Linear(768, NUM_CLASSES).to(device)
    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier,
        device_ids=[local_rank]
    )


    # -------------------------
    # Data
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    train_ds = datasets.ImageFolder("CSF_dataset/train", transform=transform)
    val_ds   = datasets.ImageFolder("CSF_dataset/val", transform=transform)
    test_ds  = datasets.ImageFolder("CSF_dataset/test", transform=transform)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)
    test_sampler  = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)


    # -------------------------
    # Feature extraction
    # -------------------------
    def extract_features(x):
        with torch.no_grad():
            feats = backbone.forward_features(x)
            return feats["x_norm_clstoken"]


    # -------------------------
    # Train loop
    # -------------------------
    def train_one_epoch(epoch):
        classifier.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0

        for imgs, labels in tqdm(train_loader, disable=(rank != 0)):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = extract_features(imgs)
            outputs = classifier(feats)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


    # -------------------------
    # Eval (DDP reduced)
    # -------------------------
    @torch.no_grad()
    def evaluate(loader):
        classifier.eval()

        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        loss_sum = torch.tensor(0.0, device=device)

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = extract_features(imgs)
            outputs = classifier(feats)

            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum()
            total += labels.size(0)
            loss_sum += loss.item()

        # ---- sync across GPUs ----
        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)

        acc = correct.item() / total.item()
        avg_loss = loss_sum.item() / dist.get_world_size()

        return acc, avg_loss


    # -------------------------
    # Training loop
    # -------------------------
    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_acc, val_loss = evaluate(val_loader)

        if rank == 0:
            print(f"\nEpoch {epoch}")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(classifier.module.state_dict(), "best_linear_probe.pth")
                print("Saved best model")


    # -------------------------
    # Test
    # -------------------------
    test_acc, test_loss = evaluate(test_loader)

    if rank == 0:
        print(f"\nTEST ACC: {test_acc:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
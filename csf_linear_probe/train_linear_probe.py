import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)
import torch
import torch.nn as nn
import torch.distributed as dist
from sympy import false
from tensorboard.plugin_util import experiment_id
from torch.utils.tensorboard import SummaryWriter
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
BATCH_SIZE = 128  # per GPU
LR = 1e-3
EPOCHS = 100
NUM_CLASSES = 15
IMG_SIZE = 224
DATASET_NAME = "CSF"
ddp_status = True
checkpoint_path = "/home/adminuser/PycharmProjects/Selfsupervised/dinoCSF/checkpoints/"
class_names = ['Artifizielle Zelle', 'Erythrophage', 'Erythrozyt', 'Hämatoidin', 'Hämosiderophage',	'Kernschatten',
                   'Lymphozyt', 'Mitose', 'Monozyt', 'Plasmazelle', 'Tumor', 'aktivierter Lymphozyt', 'aktivierter Monozyt',
                   'eosinophiler Granulozyt', 'neutrophiler Granulozyt']

best_model_path = ""
# -------------------------
# Train
# -------------------------
def main():
    if ddp_status:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #world_size = dist.get_world_size()
    rank = dist.get_rank()

    DATASET_ROOT = "/home/adminuser/backup/Neuropathological_Data/projects/Akash/CSF_classification/raw_data/zenodo/"
    SPLIT_JSON = "data/split_file.json"
    ODS_FILE = "data/supplementary_table_73.ods"

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

    if DATASET_NAME == "CSF":

        from csf_dataset import build_csf_datasets

        train_ds, val_ds, test_ds = build_csf_datasets(
            data_root=DATASET_ROOT,
            split_json_path=SPLIT_JSON,
            ods_file_path=ODS_FILE,
            transform=transform
        )

    else:

        train_ds = datasets.ImageFolder("CSF_dataset/train", transform=transform)
        val_ds = datasets.ImageFolder("CSF_dataset/val", transform=transform)
        test_ds = datasets.ImageFolder("CSF_dataset/test", transform=transform)

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
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LR,
        weight_decay=1e-4
    )


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

    @torch.no_grad()
    def evaluate_test_metrics(loader):
        classifier.eval()

        local_preds = []
        local_labels = []

        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        loss_sum = torch.tensor(0.0, device=device)

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = extract_features(imgs)
            outputs = classifier(feats)

            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum()
            total += labels.size(0)
            loss_sum += loss.item()

            local_preds.append(preds)
            local_labels.append(labels)

        # --------------------
        # Accuracy/Loss
        # --------------------
        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)

        acc = correct.item() / total.item()
        avg_loss = loss_sum.item() / dist.get_world_size()

        # --------------------
        # Gather predictions
        # --------------------
        local_preds = torch.cat(local_preds)
        local_labels = torch.cat(local_labels)

        gathered_preds = [torch.zeros_like(local_preds)
                          for _ in range(dist.get_world_size())]

        gathered_labels = [torch.zeros_like(local_labels)
                           for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_labels, local_labels)

        if rank == 0:
            y_pred = torch.cat(gathered_preds).cpu().numpy()
            y_true = torch.cat(gathered_labels).cpu().numpy()

            return acc, avg_loss, y_true, y_pred

        return acc, avg_loss, None, None

    # -------------------------
    # Training loop
    # -------------------------

    best_acc = 0.0
    best_model_path = checkpoint_path + "best_model.pth"
    # Create TensorBoard writer only on rank 0
    if rank == 0:
        os.makedirs("tensorboard_logs", exist_ok=True)
        writer = SummaryWriter(log_dir="tensorboard_logs")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_acc, val_loss = evaluate(val_loader)

        if rank == 0:
            print(f"\nEpoch {epoch}")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

            # TensorBoard logging
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)


            if val_acc > best_acc:
                best_acc = val_acc
                #best_model_path = checkpoint_path + f"checkpoint_{epoch}.pth"

                torch.save(classifier.module.state_dict(), best_model_path)

                # Log best accuracy
                writer.add_scalar("Accuracy/Best_Validation", best_acc, epoch)
                print("Saved best model: ", epoch)


    # -------------------------
    # Test
    # -------------------------
    dist.barrier()

    classifier.module.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    test_acc, test_loss, y_true, y_pred = evaluate_test_metrics(test_loader)

    if rank == 0:
        #print(f"\nTEST ACC: {test_acc:.4f}")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names
        )

        print("\nConfusion Matrix:")
        print(cm_df)

        precision, recall, f1, support = \
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                zero_division=0
            )

        '''
        print("\nClass-wise Metrics")
        print("-" * 100)

        for i, cls in enumerate(class_names):
            print(
                f"{cls:30s} | "
                f"Recall(Sens): {recall[i]:.4f} | "
                f"Precision: {precision[i]:.4f} | "
                f"F1: {f1[i]:.4f} | "
                f"N={support[i]}"
            )
        '''
        if rank == 0:

            print("\nClass-wise Sensitivity / Specificity / F1")
            print("-" * 120)

            for i, cls in enumerate(class_names):
                TP = cm[i, i]

                FN = cm[i, :].sum() - TP

                FP = cm[:, i].sum() - TP

                TN = cm.sum() - TP - FN - FP

                sensitivity = TP / (TP + FN + 1e-12)

                specificity = TN / (TN + FP + 1e-12)

                f1_score = f1[i]

                print(
                    f"{cls:30s} | "
                    f"Sens: {sensitivity:.4f} | "
                    f"Spec: {specificity:.4f} | "
                    f"F1: {f1_score:.4f}"
                )

        # TensorBoard logging
        writer.add_scalar("Loss/Test", test_loss, EPOCHS)
        writer.add_scalar("Accuracy/Test", test_acc, EPOCHS)

        writer.flush()
        writer.close()


    cleanup_ddp()


if __name__ == "__main__":
    main()
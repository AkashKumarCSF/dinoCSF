import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.ssl_dataset import SSLImageFolder
from models.dino_ssl import DINOStudent, DINOTeacher
import os
from tqdm import tqdm
import time


import torch.nn.functional as F

def dino_loss(student_out, teacher_out, temp_s=0.1, temp_t=0.04):
    student_out = F.normalize(student_out, dim=-1)
    teacher_out = F.normalize(teacher_out, dim=-1)

    student_logits = student_out / temp_s
    teacher_logits = teacher_out.detach() / temp_t

    loss = -(teacher_logits.softmax(dim=-1) * student_logits.log_softmax(dim=-1)).sum(dim=-1)
    return loss.mean()


def freeze_last_4_blocks_torchvision(model):

    # 1. Freeze everything
    for p in model.backbone.parameters():
        p.requires_grad = False

    # 2. Get transformer blocks correctly (THIS is key)
    blocks = model.backbone.encoder.layers  # Sequential

    assert len(blocks) == 12, f"Expected 12 blocks, got {len(blocks)}"

    # 3. Unfreeze last 4 blocks using indexing
    for block in blocks[-4:]:
        for p in block.parameters():
            p.requires_grad = True

    # 4. Always unfreeze final LayerNorm
    for p in model.backbone.encoder.ln.parameters():
        p.requires_grad = True

    # 5. Always train projection head
    for p in model.projector.parameters():
        p.requires_grad = True

    print("\n[INFO] Frozen backbone except last 4 transformer blocks")
    for name, p in model.backbone.named_parameters():
        if p.requires_grad:
            print(name)


def freeze_last_4_blocks_dinov2(model):

    # 1. Freeze everything in backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # 2. Unfreeze last 4 transformer blocks
    for block in model.backbone.blocks[-4:]:
        for p in block.parameters():
            p.requires_grad = True

    # 3. Always unfreeze final norm
    for p in model.backbone.norm.parameters():
        p.requires_grad = True

    # 4. Train projection head
    for p in model.projector.parameters():
        p.requires_grad = True

    print("[INFO] Unfroze last 4 DINOv2 blocks + norm + projector")


def print_trainable_params(model):
    print("\n=== TRAINABLE PARAMETERS ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)


def count_trainable_params(model):
    total, trainable = 0, 0

    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print("\n[MODEL STATS]")
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    print(f"Total params: {total / 1e6:.2f}M")


def inspect(model):
    print("\n=== CHILD MODULES ===")
    for name, module in model.backbone.named_children():
        print(name, type(module))

    print("\n=== PARAMETER NAMES (first 50) ===")
    for i, (name, _) in enumerate(model.backbone.named_parameters()):
        print(name)
        if i > 50:
            break


def train_ssl(
    dataset_path,
    epochs=100,
    batch_size=64,
    lr=1e-4,
    save_dir="ssl_checkpoints"
):

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="ssl_tensorboard/dino_ssl")

    dataset = SSLImageFolder(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    student = DINOStudent().cuda()
    teacher = DINOTeacher(student).cuda()


    # IMPORTANT: freeze BEFORE optimizer
    freeze_last_4_blocks_dinov2(student)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
        weight_decay=0.04
    )

    best_loss = float("inf")
    patience_counter = 0

    #print("\n=== STUDENT MODEL ===")
    #print(student)

    for epoch in range(epochs):

        student.train()
        teacher.teacher.eval()

        total_loss = 0
        start_epoch = time.time()

        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, (x1, x2) in enumerate(loader_tqdm):
            step_start = time.time()

            x1, x2 = x1.cuda(), x2.cuda()

            s1 = student(x1)
            s2 = student(x2)

            with torch.no_grad():
                t1 = teacher(x1)
                t2 = teacher(x2)

            loss = dino_loss(s1, t2) + dino_loss(s2, t1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            teacher.update(student)

            total_loss += loss.item()

            # ---- LIVE LOGGING ----
            loader_tqdm.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / (step + 1):.4f}",
                "step_time": f"{time.time() - step_start:.2f}s"
            })

        avg_loss = total_loss / len(loader)

        epoch_time = time.time() - start_epoch

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")

        #print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)

        torch.save(student.state_dict(),
                   os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

        # best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(student.state_dict(),
                       os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("Early stopping triggered")
            break

    writer.close()

    return os.path.join(save_dir, "best_model.pt")
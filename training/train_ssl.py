import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.ssl_dataset import SSLImageFolder
from models.dino_ssl import DINOStudent, DINOTeacher
import os
from tqdm import tqdm
import time


import torch.nn.functional as F
from torch import nn


class DINOCenter(nn.Module):
    def __init__(self, out_dim=256, momentum=0.995):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.momentum).add_(batch_center, alpha=1 - self.momentum)


def dino_loss(
    student_out,
    teacher_out,
    center,
    temp_s=0.1,
    temp_t=0.04
):

    student_out = F.normalize(
        student_out,
        dim=-1
    )

    teacher_out = F.normalize(
        teacher_out,
        dim=-1
    )

    student_logits = student_out / temp_s

    teacher_logits = (
        teacher_out.detach() - center
    ) / temp_t

    teacher_probs = F.softmax(
        teacher_logits,
        dim=-1
    )

    student_log_probs = F.log_softmax(
        student_logits,
        dim=-1
    )

    loss = -(
        teacher_probs * student_log_probs
    ).sum(dim=-1)

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
        p.requires_grad = True
    print("All parameters are trainable")

    # 2. Unfreeze last 4 transformer blocks
    '''
    for block in model.backbone.blocks[-4:]:
        for p in block.parameters():
            p.requires_grad = True
    print("[INFO] Unfroze last 4 DINOv2 blocks + norm + projector")
    '''
    # 3. Always unfreeze final norm
    for p in model.backbone.norm.parameters():
        p.requires_grad = True

    # 4. Train projection head
    for p in model.projector.parameters():
        p.requires_grad = True




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
    epochs=50,
    batch_size=64,
    lr=1e-4,
    save_dir="ssl_checkpoints",
    resume=False,
    checkpoint_path = None,
    log_dir = "ssl_tensorboard/dino_ssl/"
):

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    dataset = SSLImageFolder(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:0")
    student = DINOStudent().to(device)
    teacher = DINOTeacher(student).to(device)
    centering = DINOCenter(out_dim=256).to(device)

    # IMPORTANT: freeze BEFORE optimizer
    freeze_last_4_blocks_dinov2(student)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
        weight_decay=0.04
    )

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if resume:

        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Backward compatibility with old checkpoints
        if "student" in checkpoint:
            student.load_state_dict(checkpoint["student"])
            teacher.load_state_dict(checkpoint["teacher"])
            centering.load_state_dict(checkpoint["center"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint.get("best_loss", float("inf"))

        else:
            # Old checkpoint containing only student weights
            student.load_state_dict(checkpoint)

            # Teacher is initialized from the loaded student
            teacher = DINOTeacher(student).cuda()

            # Extract epoch number from filename
            filename = os.path.basename(checkpoint_path)
            start_epoch = int(filename.split("_")[-1].split(".")[0])

        print(f"\nResuming training from epoch {start_epoch + 1}")


    #print("\n=== STUDENT MODEL ===")
    #print(student)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs):

        student.train()
        teacher.teacher.eval()

        total_loss = 0
        start_epoch = time.time()

        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        #for step, (x1, x2) in enumerate(loader_tqdm):
        for step, views in enumerate(loader_tqdm):
            step_start = time.time()

            views = [v.cuda(non_blocking=True) for v in views]

            global_views = views[:2]
            local_views = views[2:]
            student_views = views  # all crops

            # -------------------------
            # TEACHER FORWARD (NO GRAD)
            # -------------------------
            with torch.no_grad():
                teacher_outs = [teacher(v) for v in global_views]
                teacher_cat = torch.cat(teacher_outs, dim=0).detach()

                centering.update(teacher_cat)

            # -------------------------
            # STUDENT FORWARD (WITH GRAD)
            # -------------------------
            with amp.autocast():
                student_outs = [student(v) for v in student_views]

                loss = 0.0
                n_terms = 0

                for t in teacher_outs:
                    for s in student_outs:
                        loss += dino_loss(
                            s,
                            t,
                            centering.center
                        )
                        n_terms += 1

                loss = loss / n_terms

            # -------------------------
            # BACKPROP
            # -------------------------
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # -------------------------
            # EMA UPDATE
            # -------------------------
            teacher.update(student)

            total_loss += loss.item()

            loader_tqdm.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / (step + 1):.4f}",
            })

        avg_loss = total_loss / len(loader)

        epoch_time = time.time() - start_epoch

        with torch.no_grad():
            x = next(iter(loader))[0].cuda()

            z = student(x)

            print(
                "mean:", z.mean().item(),
                "std:", z.std().item()
            )

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")

        #print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        if (epoch + 1) % 15 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "center": centering.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            )

        # best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "center": centering.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(save_dir, "best_model.pt")
            )
        else:
            patience_counter += 1
        '''
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
        '''

    writer.close()

    return os.path.join(save_dir, "best_model.pt")
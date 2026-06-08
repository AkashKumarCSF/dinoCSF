import os
import yaml
import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from training.ddp import setup_ddp, cleanup_ddp
from training.trainer import Trainer
from training.evaluator import Evaluator

from datasets.dataloader import build_dataloaders

from transforms.transforms import get_transforms

from models.backbones import build_backbone
from models.feature_extractor import FeatureExtractor
from models.probe import LinearProbe

from utils.metrics import compute_metrics


# ----------------------------------
# Config loader
# ----------------------------------
def load_config(config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ----------------------------------
# Main
# ----------------------------------
def main():

    cfg = load_config(
        "configs/config.yaml"
    )

    # ----------------------------------
    # DDP
    # ----------------------------------
    if cfg["system"]["ddp"]:

        local_rank = setup_ddp()

        device = torch.device(
            f"cuda:{local_rank}"
        )

        rank = dist.get_rank()

    else:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        rank = 0

        local_rank = 0

    # ----------------------------------
    # Transforms
    # ----------------------------------
    transform = get_transforms(
        cfg["data"]["img_size"]
    )

    # ----------------------------------
    # Datasets
    # ----------------------------------
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler
    ) = build_dataloaders(
        cfg,
        transform
    )

    # ----------------------------------
    # Backbone
    # ----------------------------------
    backbone, embed_dim = build_backbone(
        cfg,
        device
    )

    feature_extractor = FeatureExtractor(
        backbone,
        cfg["model"]["backbone"]
    )

    # ----------------------------------
    # Probe
    # ----------------------------------
    classifier = LinearProbe(
        embed_dim,
        cfg["classes"]["num_classes"]
    ).to(device)

    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier,
        device_ids=[local_rank]
    )

    # ----------------------------------
    # Trainer
    # ----------------------------------
    trainer = Trainer(
        classifier=classifier,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        train_sampler=train_sampler,
        device=device,
        rank=rank,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # ----------------------------------
    # Evaluator
    # ----------------------------------
    evaluator = Evaluator(
        classifier=classifier,
        feature_extractor=feature_extractor,
        criterion=trainer.criterion,
        device=device,
        rank=rank
    )

    # ----------------------------------
    # TensorBoard
    # ----------------------------------
    if rank == 0:

        os.makedirs(
            cfg["output"]["tensorboard_dir"],
            exist_ok=True
        )

        writer = SummaryWriter(
            log_dir=cfg["output"]["tensorboard_dir"]
        )

    # ----------------------------------
    # Checkpoint
    # ----------------------------------
    os.makedirs(
        cfg["output"]["checkpoint_dir"],
        exist_ok=True
    )

    best_model_path = os.path.join(
        cfg["output"]["checkpoint_dir"],
        "best_model.pth"
    )

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    best_acc = 0.0

    for epoch in range(
            cfg["training"]["epochs"]
    ):

        train_loss = trainer.train_one_epoch(
            epoch
        )

        val_acc, val_loss = evaluator.evaluate(
            val_loader
        )

        if rank == 0:

            print(
                f"\nEpoch {epoch}"
            )

            print(
                f"Train loss: {train_loss:.4f}"
            )

            print(
                f"Val loss: {val_loss:.4f}, "
                f"Val acc: {val_acc:.4f}"
            )

            writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch
            )

            writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch
            )

            writer.add_scalar(
                "Accuracy/Validation",
                val_acc,
                epoch
            )

            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    classifier.module.state_dict(),
                    best_model_path
                )

                writer.add_scalar(
                    "Accuracy/Best_Validation",
                    best_acc,
                    epoch
                )

                print(
                    f"Saved best model at epoch {epoch}"
                )

    # ----------------------------------
    # Load Best Model
    # ----------------------------------
    dist.barrier()

    classifier.module.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    # ----------------------------------
    # Test
    # ----------------------------------
    (
        test_acc,
        test_loss,
        y_true,
        y_pred
    ) = evaluator.evaluate_test_metrics(
        test_loader
    )

    # ----------------------------------
    # Metrics
    # ----------------------------------
    if rank == 0:

        class_names = cfg["classes"]["names"]

        df, macro, cm = compute_metrics(
            y_true,
            y_pred,
            class_names
        )

        print("\nTEST SET RESULTS (Best Model)")
        print("=" * 80)

        print("\nConfusion Matrix:\n")
        print(cm_df.to_string())
        print("=" * 80)

        print("\nPer-class metrics:\n")
        print(df.to_string(index=False))


        for m in class_metrics:

            print(
                f"{m['class']:30s} | "
                f"Sens: {m['sensitivity']:.4f} | "
                f"Spec: {m['specificity']:.4f} | "
                f"F1: {m['f1']:.4f}"
            )

        print("\nMacro Averages:")
        print("-" * 40)

        print(f"Precision   : {macro['Precision']:.4f}")
        print(f"Recall      : {macro['Recall']:.4f}")
        print(f"F1-score    : {macro['F1']:.4f}")
        print(f"Specificity : {macro['Specificity']:.4f}")

        writer.add_scalar(
            "Loss/Test",
            test_loss,
            cfg["training"]["epochs"]
        )

        writer.add_scalar(
            "Accuracy/Test",
            test_acc,
            cfg["training"]["epochs"]
        )

        writer.flush()
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
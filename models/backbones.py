import torch
import torchvision.models as models


def build_backbone(cfg, device):

    backbone_name = cfg["model"]["backbone"]

    if backbone_name == "dinov2":

        model = torch.hub.load(
            "facebookresearch/dinov2",
            cfg["model"]["dinov2"]["name"]
        )

        embed_dim = cfg["model"]["dinov2"]["embedding_dim"]

    elif backbone_name == "mae":

        from models.mae_loader import load_mae

        model = load_mae(
            cfg["model"]["mae"]["checkpoint"]
        )

        embed_dim = cfg["model"]["mae"]["embedding_dim"]

    elif backbone_name == "vit":
        model = models.vit_b_16(weights=None)  # IMPORTANT: no ImageNet weights

        embed_dim = 768

        ckpt_path = cfg["model"]["vit"]["checkpoint"]

        if ckpt_path is not None:
            print(f"\nLoading SSL-trained ViT checkpoint: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            new_ckpt = {}

            for k, v in ckpt.items():

                if k.startswith("backbone."):
                    new_k = k.replace("backbone.", "")
                    new_ckpt[new_k] = v

            missing, unexpected = model.load_state_dict(new_ckpt, strict=False)

            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")

        model.to(device)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}"
        )

    model.to(device)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, embed_dim
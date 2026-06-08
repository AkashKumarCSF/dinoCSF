import torch


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

    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}"
        )

    model.to(device)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, embed_dim
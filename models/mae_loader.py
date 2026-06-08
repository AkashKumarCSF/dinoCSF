def load_mae(checkpoint_path):

    model = mae_vit_base_patch16()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=False
    )

    return model
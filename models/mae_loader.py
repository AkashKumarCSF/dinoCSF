import torch
from timm.models.vision_transformer import vit_base_patch16_224


def load_mae(checkpoint_path):

    # ---- 1. Import MAE model definition ----
    # This comes from the official MAE repo (timm-based)
    import timm


    # ---- 2. Create model architecture ----
    model = vit_base_patch16_224(
        pretrained=False
    )

    # MAE models use "encoder-only" weights during finetuning/probing
    model.fc = torch.nn.Identity()

    # ---- 3. Load checkpoint ----
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    # MAE checkpoints usually store weights under "model"
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # ---- 4. Clean keys (important for MAE checkpoints) ----
    new_state_dict = {}

    for k, v in state_dict.items():

        # remove "module." if multi-GPU trained
        k = k.replace("module.", "")

        # keep only encoder weights (important for MAE)
        if k.startswith("encoder"):

            k = k.replace("encoder.", "")

            new_state_dict[k] = v

    # ---- 5. Load weights ----
    model.load_state_dict(
        new_state_dict,
        strict=False
    )

    return model
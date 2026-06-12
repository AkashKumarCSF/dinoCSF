import torch


class FeatureExtractor:

    def __init__(self, backbone, backbone_name):

        self.backbone = backbone
        self.backbone_name = backbone_name

    @torch.no_grad()
    def __call__(self, x):

        if self.backbone_name == "dinov2":

            feats = self.backbone.forward_features(x)

            return feats["x_norm_patchtokens"].mean(dim=1) #feats["x_norm_clstoken"]



        elif self.backbone_name == "mae":

            feats = self.backbone.forward_features(x)

            cls = feats[:, 0]  # CLS token

            patch = feats[:, 1:].mean(dim=1)  # mean of patch tokens

            return torch.cat([cls, patch], dim=-1)

        elif self.backbone_name == "vit":

            # torchvision ViT returns logits normally, so we bypass heads
            feats = self.backbone._process_input(x)

            # forward manually through encoder
            batch_class_token = self.backbone.class_token.expand(x.shape[0], -1, -1)

            x = self.backbone.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)

            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)

            cls_token = x[:, 0]

            patch_tokens = x[:, 1:].mean(dim=1)

            return cls_token + patch_tokens  # or just cls_token

        else:
            raise NotImplementedError
import torch


class FeatureExtractor:

    def __init__(self, backbone, backbone_name):

        self.backbone = backbone
        self.backbone_name = backbone_name

    @torch.no_grad()
    def __call__(self, x):

        if self.backbone_name == "dinov2":

            feats = self.backbone.forward_features(x)

            return feats["x_norm_clstoken"]

        elif self.backbone_name == "mae":

            feats = self.backbone.forward_features(x)

            return feats[:, 0]

        else:
            raise NotImplementedError
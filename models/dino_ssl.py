import torch
import torch.nn as nn
import torchvision.models as models

class DINOStudent(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = models.vit_b_16(weights="IMAGENET1K_V1")

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14'
        )
        #self.backbone.heads = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(768, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 256, bias=False),
        )

        # Weight-normalized last layer (same idea as DINO)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(256, 256, bias=False)
        )

        # Initialize scale to 1
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)

        x = self.projector(features)

        x = F.normalize(x, dim=-1)

        x = self.last_layer(x)

        return x


class DINOTeacher(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.teacher = DINOStudent()
        self.teacher.load_state_dict(student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student, momentum=0.999):
        for ps, pt in zip(student.parameters(), self.teacher.parameters()):
            pt.data = pt.data * momentum + ps.data * (1 - momentum)

    def forward(self, x):
        return self.teacher(x)
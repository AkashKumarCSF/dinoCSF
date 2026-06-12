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
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)


class DINOTeacher(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.teacher = DINOStudent()
        self.teacher.load_state_dict(student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student, momentum=0.996):
        for ps, pt in zip(student.parameters(), self.teacher.parameters()):
            pt.data = pt.data * momentum + ps.data * (1 - momentum)

    def forward(self, x):
        return self.teacher(x)
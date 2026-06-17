import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SSLImageFolder(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".png"):
                    self.paths.append(os.path.join(root, f))

        self.global_transform = T.Compose([
            T.RandomResizedCrop(
                224,
                scale=(0.5, 1.0)
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                0.4,
                0.4,
                0.4,
                0.1
            ),
            T.GaussianBlur(5),
            T.ToTensor(),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                112,
                scale=(0.1, 0.4)
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                0.4,
                0.4,
                0.4,
                0.1
            ),
            T.GaussianBlur(5),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        global1 = self.global_transform(img)
        global2 = self.global_transform(img)

        local1 = self.local_transform(img)
        local2 = self.local_transform(img)
        local3 = self.local_transform(img)
        local4 = self.local_transform(img)

        return [
            global1,
            global2,
            local1,
            local2,
            local3,
            local4
        ]
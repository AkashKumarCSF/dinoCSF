import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SSLImageFolder(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
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
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
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
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        while True:
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                break
            except (OSError, IOError) as e:
                print(f"Skipping corrupted image: {self.paths[idx]} ({e})")

        global1 = self.global_transform(img)
        global2 = self.global_transform(img)

        # Using only 2 global crops similar to C Matek, as local cropping is interfering with cell morphology

        #local1 = self.local_transform(img)
        #local2 = self.local_transform(img)
        #local3 = self.local_transform(img)
        #local4 = self.local_transform(img)

        return [
            global1,
            global2
        ]
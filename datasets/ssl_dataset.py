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

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=5),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # two augmented views (DINO-style)
        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2
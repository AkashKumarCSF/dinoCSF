from torchvision import transforms


def get_transforms(img_size):
    color_jitter = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.5,
    }
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(**color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
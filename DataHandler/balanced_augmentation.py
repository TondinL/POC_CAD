import random
from torchvision import transforms
from spatial_transform import (
    Compose, Resize, RandomHorizontalFlip, RandomResizedCrop,
    ColorJitter, ToTensor, Normalize
)

class BalancedAugmentation:
    """
    Applica trasformazioni diverse a seconda della classe:
      - class_idx 0: leggera
      - class_idx 1: intensiva
    """
    def __init__(self, image_size=224):
        # pipeline leggera (es. per NORMAL)
        self.light_aug = Compose([
            Resize(256),
            RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        # pipeline forte (es. per ABNORMAL)
        self.strong_aug = Compose([
            Resize(256),
            RandomResizedCrop(image_size, scale=(0.5, 1.0)),  # crop più aggressivo
            RandomHorizontalFlip(p=0.7),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img, class_idx):
        """
        img: PIL.Image
        class_idx: int (0=Normal, 1=Abnormal)
        """
        if class_idx == 1:  # ABNORMAL
            self.strong_aug.randomize_parameters()
            return self.strong_aug(img)
        else:
            self.light_aug.randomize_parameters()
            return self.light_aug(img)

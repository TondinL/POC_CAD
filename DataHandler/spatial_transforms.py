# DataHandler/spatial_transforms.py
from typing import Sequence, Optional
from torchvision import transforms as T

class Compose:
    """
    Compose che supporta randomize_parameters() se presente nei sotto-trasform.
    """
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            if hasattr(t, "randomize_parameters"):
                t.randomize_parameters()


class Resize:
    def __init__(self, size):
        self.transform = T.Resize(size)

    def __call__(self, img):
        return self.transform(img)

    def randomize_parameters(self):
        pass


class RandomResizedCrop:
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3/4, 4/3)):
        self.transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    def __call__(self, img):
        return self.transform(img)

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.transform = T.RandomHorizontalFlip(p=p)

    def __call__(self, img):
        return self.transform(img)

    def randomize_parameters(self):
        pass


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img):
        return self.transform(img)

    def randomize_parameters(self):
        pass


class ToTensor:
    def __init__(self):
        self.transform = T.ToTensor()

    def __call__(self, img):
        return self.transform(img)

    def randomize_parameters(self):
        pass


class Normalize:
    def __init__(self, mean, std):
        self.transform = T.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        return self.transform(tensor)

    def randomize_parameters(self):
        pass

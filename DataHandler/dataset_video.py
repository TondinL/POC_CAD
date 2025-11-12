# dataset_video.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class VideoDataset(Dataset):
    """
    Legge frame da cartelle organizzate come:
    root/class_name/video_name/*.jpg
    Applica trasformazioni temporali e spaziali definite nei moduli dedicati.
    """

    def __init__(self, root_path, spatial_transform=None, temporal_transform=None):
        self.root_path = Path(root_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        # Legge le classi (es: NORMAL, ABNORMAL)
        found = [d.name for d in self.root_path.iterdir() if d.is_dir()]
        # Enforce ordine canonico: NORMAL=0, ABNORMAL=1
        expected = ["NORMAL", "ABNORMAL"]
        self.classes = [c for c in expected if c in found]
        if set(self.classes) != set(expected):
            raise RuntimeError(f"Mi aspetto le classi {expected}, trovate {sorted(found)}")

        self.class_to_idx = {"NORMAL": 0, "ABNORMAL": 1}

        # Costruisce la lista (frame_paths, label)
        self.samples = []
        for cls_name in self.classes:
            cls_folder = self.root_path / cls_name
            for video_dir in sorted(cls_folder.iterdir()):
                if not video_dir.is_dir():
                    continue
                frame_paths = sorted(video_dir.glob("*.jpg"))
                if len(frame_paths) == 0:
                    continue
                self.samples.append((frame_paths, self.class_to_idx[cls_name]))

        print(f"Trovati {len(self.samples)} video in {len(self.classes)} classi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frame_indices = list(range(len(frame_paths)))

        # --- Trasformazioni temporali ---
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        # --- Caricamento frame ---
        frames = [Image.open(frame_paths[i]).convert('RGB') for i in frame_indices]

        # --- Trasformazioni spaziali ---
        if self.spatial_transform is not None:
            # se è un BalancedAugmentation
            if hasattr(self.spatial_transform, '__call__') and \
               'class_idx' in self.spatial_transform.__call__.__code__.co_varnames:
                self.spatial_transform.light_aug.randomize_parameters()
                self.spatial_transform.strong_aug.randomize_parameters()
                frames = [self.spatial_transform(img, label) for img in frames]
            else:
                self.spatial_transform.randomize_parameters()
                frames = [self.spatial_transform(img) for img in frames]

        # converte in tensore [T, C, H, W]
        frames_tensor = torch.stack(frames)
        return frames_tensor, label

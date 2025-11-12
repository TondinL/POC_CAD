# analyze_dataset_augmented.py
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# aggiungi la cartella principale del progetto al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataHandler.dataset_video import VideoDataset
from DataHandler.temporal_transforms import TemporalRandomCropStrict
from DataHandler.balanced_augmentation import BalancedAugmentation

ROOT = Path("../Dataset_jpg")
OUT_DIR = Path("Analysis_augmented")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_frames_post_augmentation(dataset):
    """Conta quanti frame totali vengono generati per classe dopo le trasformazioni"""
    frame_counts = {0: 0, 1: 0}
    for i in tqdm(range(len(dataset)), desc="Analizzando video"):
        frames, label = dataset[i]
        frame_counts[label] += frames.shape[0]  # frames è [T, C, H, W]
    return frame_counts

def plot_results(frame_counts, out_dir):
    labels = ["NORMAL", "ABNORMAL"]
    values = [frame_counts[0], frame_counts[1]]
    plt.bar(labels, values, color=["skyblue", "salmon"])
    plt.title("Numero totale di frame post-augmentation")
    plt.ylabel("Frame totali")
    plt.tight_layout()
    plt.savefig(out_dir / "post_augmentation_frame_distribution.png", dpi=200)
    plt.close()

def main():
    temporal = TemporalRandomCropStrict(size=8)
    spatial = BalancedAugmentation(image_size=224)

    dataset = VideoDataset(ROOT, spatial_transform=spatial, temporal_transform=temporal)
    print(f"[INFO] Dataset con augmentation: {len(dataset)} video")

    frame_counts = analyze_frames_post_augmentation(dataset)
    print(f"[INFO] Frame totali post-augmentation: {frame_counts}")

    plot_results(frame_counts, OUT_DIR)

    # salva in JSON
    import json
    with open(OUT_DIR / "post_augmentation_frame_stats.json", "w", encoding="utf-8") as f:
        json.dump({"frame_counts": frame_counts}, f, indent=2)
    print(f"[OK] Analisi completata. File salvati in {OUT_DIR}")

if __name__ == "__main__":
    main()
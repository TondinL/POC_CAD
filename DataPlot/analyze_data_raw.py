# analyze_dataset_raw.py
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

ROOT = Path("../Dataset_jpg")
OUT_DIR = Path("Analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_raw_dataset(root):
    class_stats = {}
    total_videos = 0

    for cls_name in sorted(os.listdir(root)):
        cls_path = root / cls_name
        if not cls_path.is_dir():
            continue

        n_videos = 0
        frame_counts = []
        frame_sizes = []

        for video_dir in cls_path.iterdir():
            if not video_dir.is_dir():
                continue
            frames = list(video_dir.glob("*.jpg"))
            if not frames:
                continue

            n_videos += 1
            frame_counts.append(len(frames))

            # prendi il primo frame per le dimensioni
            try:
                with Image.open(frames[0]) as im:
                    frame_sizes.append(im.size)
            except Exception:
                continue

        total_videos += n_videos
        class_stats[cls_name] = {
            "videos": int(n_videos),
            "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
            "min_frames": int(np.min(frame_counts)) if frame_counts else 0,
            "max_frames": int(np.max(frame_counts)) if frame_counts else 0,
            "avg_size": float(np.mean([w*h for (w,h) in frame_sizes])) if frame_sizes else 0.0
        }

    return {"total_videos": int(total_videos), "classes": class_stats}


def plot_distributions(stats, out_dir):
    cls_names = list(stats["classes"].keys())
    n_videos = [stats["classes"][c]["videos"] for c in cls_names]
    avg_frames = [stats["classes"][c]["avg_frames"] for c in cls_names]

    plt.figure()
    plt.bar(cls_names, n_videos, color=["skyblue","salmon"])
    plt.title("Numero di video per classe")
    plt.ylabel("Conteggio")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(cls_names, avg_frames, color=["skyblue","salmon"])
    plt.title("Durata media (frame per video)")
    plt.ylabel("Frame medi")
    plt.tight_layout()
    plt.savefig(out_dir / "frame_distribution.png", dpi=200)
    plt.close()


def main():
    stats = analyze_raw_dataset(ROOT)
    # conversione a tipi base garantita
    with open(OUT_DIR / "dataset_raw_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    plot_distributions(stats, OUT_DIR)
    print(f"[OK] Analisi completata. Grafici salvati in {OUT_DIR}")


if __name__ == "__main__":
    main()
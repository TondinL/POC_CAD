# src/app/runtime.py
from pathlib import Path
import tempfile
import subprocess
import shutil

import torch
import torch.nn.functional as F
from PIL import Image

# riuso le tue trasformazioni
from DataHandler.spatial_transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize
)

from model import build_model


def temporal_center_indices(n_frames: int, T: int):
    if n_frames <= 0:
        return [0]*T
    if n_frames >= T:
        import numpy as np
        idx = np.linspace(0, n_frames - 1, num=T)
        return list(np.round(idx).astype(int))
    else:
        idx = list(range(n_frames))
        idx += [n_frames - 1] * (T - n_frames)
        return idx


class InferenceEngine:
    def __init__(self, ckpt_path: str, depth: int = 50, device: str = "cpu",
                 temporal_size: int = 8, image_size: int = 224, threshold: float = 0.5):
        self.device = torch.device(device)
        self.temporal_size = temporal_size
        self.image_size = image_size
        self.threshold = threshold
        self.class_names = ("NORMAL", "ABNORMAL")

        # modello “vuoto” + carico checkpoint fine-tunato
        self.model = build_model(
            model_depth=depth,
            num_classes=2,
            pretrained_path=None,
            freeze_until=None,
            device=device,
        )
        self._load_checkpoint(ckpt_path)
        self.model.eval()

        # trasformazioni deterministiche
        self.transform = Compose([
            Resize(256),
            CenterCrop(self.image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

    def _load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print("[WARN] missing keys:", list(missing))
        if unexpected:
            print("[WARN] unexpected keys:", list(unexpected))

    @torch.no_grad()
    def predict_video(self, video_path: str):
        """
        1) Estrae frame jpeg via ffmpeg in una tempdir
        2) Seleziona T frame (center/edge-pad)
        3) Applica transform e fa forward
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
        try:
            # estrai 1 frame ogni N (qui every frame; regola -vf fps= se vuoi)
            # Nota: manteniamo ordine numerico con pattern a 6 cifre
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-qscale:v", "2",  # qualità jpeg
                str(tmp_dir / "frame_%06d.jpg")
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            frames = sorted(tmp_dir.glob("frame_*.jpg"))
            n = len(frames)
            if n == 0:
                return {"ok": False, "error": "Nessun frame estratto"}

            idxs = temporal_center_indices(n, self.temporal_size)
            imgs = []
            for i in idxs:
                img = Image.open(frames[i]).convert("RGB")
                imgs.append(self.transform(img))
            tensor = torch.stack(imgs, dim=0)  # [T,C,H,W]
            tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(self.device)  # [1,C,T,H,W]

            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]  # [2]
            p_abn = float(probs[1].item())
            pred = int(p_abn >= self.threshold)

            return {
                "ok": True,
                "probabilities": {
                    "NORMAL": float(probs[0].item()),
                    "ABNORMAL": p_abn
                },
                "pred_label": pred,
                "pred_name": self.class_names[pred],
                "frames": n
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

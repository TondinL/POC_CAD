# model_viz.py
import os
from pathlib import Path
import torch
from torch import nn
import sys

# --- per il grafo ---
# pip install torchviz graphviz torchinfo
from torchviz import make_dot
from torchinfo import summary

# aggiungi la cartella principale del progetto al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import build_model  # usa la tua build_model

OUT_DIR = Path("ModelVisualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def count_frozen(model: nn.Module):
    total, frozen, trainable = 0, 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return total, frozen, trainable

def list_trainable_layers(model: nn.Module, max_show=30):
    names = [n for (n, p) in model.named_parameters() if p.requires_grad]
    head = names[:max_show]
    more = len(names) - len(head)
    return head, more

def main():
    device = torch.device("cpu")  # grafo e summary su CPU per sicurezza
    # parametri “tipici” del tuo setup
    model = build_model(
        model_depth=50,
        num_classes=2,
        pretrained_path="../PreTrained_Models/r3d50_K_200ep.pth",
        freeze_until="layer3",
        device="cpu",
    )
    model.eval()

    # dummy clip (B=1, C=3, T=8, H=W=224)
    x = torch.randn(1, 3, 8, 224, 224)

    # ----- SUMMARY (torchinfo) -----
    s = summary(model, input_size=(1, 3, 8, 224, 224), verbose=0)
    with open(OUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(s))

    # ----- FROZEN vs TRAINABLE -----
    total, frozen, trainable = count_frozen(model)
    lines = []
    lines.append(f"Parametri totali:   {total:,}")
    lines.append(f"Parametri frozen:   {frozen:,}")
    lines.append(f"Parametri trainable:{trainable:,}")
    head, more = list_trainable_layers(model)
    lines.append("\nLayer trainabili (prime voci):")
    for n in head:
        lines.append(f"  - {n}")
    if more > 0:
        lines.append(f"... (+{more} altri)")

    with open(OUT_DIR / "model_frozen_vs_trainable.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ----- GRAFO (torchviz) -----
    # Nota: serve graphviz installato nel sistema
    y = model(x)
    dot = make_dot(y, params=dict(list(model.named_parameters())))
    dot.format = "pdf"  # puoi fare anche 'png'
    out_file = OUT_DIR / "model_graph"
    dot.render(out_file.as_posix(), cleanup=True)

    print(f"[OK] Salvati:\n - {OUT_DIR/'model_summary.txt'}\n - {OUT_DIR/'model_frozen_vs_trainable.txt'}\n - {out_file.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()

# debug_pipeline.py
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from DataHandler.dataloader import make_dataloaders
from model import build_model


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


@torch.no_grad()
def eval_full(model, loader, device, class_names=("NORMAL", "ABNORMAL")):
    model.eval()
    CE = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total, correct = 0.0, 0, 0
    all_preds, all_targets = [], []
    for frames, targets in loader:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().to(device)  # B,C,T,H,W
        targets = targets.to(device)
        logits = model(frames)
        loss = CE(logits, targets)
        total_loss += loss.item()
        total += targets.size(0)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    if not all_preds:
        return {"loss": 0.0, "acc": 0.0, "cm": torch.zeros(2, 2, dtype=torch.long)}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = torch.zeros(2, 2, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return {"loss": avg_loss, "acc": acc, "cm": cm, "y_true": all_targets, "y_pred": all_preds}


def plot_cm(cm, class_names, out_png, normalize=False, title="Confusion Matrix"):
    import numpy as np
    cm_np = cm.numpy() if torch.is_tensor(cm) else cm
    if normalize:
        with np.errstate(all='ignore'):
            cm_norm = cm_np / cm_np.sum(axis=1, keepdims=True)
            cm_np = np.nan_to_num(cm_norm)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm_np, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm_np[i, j], fmt), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Debug pipeline con artefatti salvati")
    ap.add_argument("--root", type=str, default="Dataset_jpg")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--temporal-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--pretrained", type=str, default="PreTrained_Models/r3d50_K_200ep.pth")
    ap.add_argument("--freeze-until", type=str, default="layer3")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--outdir", type=str, default="FineTuned_Models")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = ensure_dir(Path(args.outdir) / f"_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"[INFO] Debug artifacts → {run_dir.resolve()}")

    # 1) dataloaders (test=0 gestito dal tuo make_dataloaders)
    train_loader, val_loader, _ = make_dataloaders(
        root_path=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_size=args.temporal_size,
        image_size=args.image_size,
        balance=(not args.no_balance),
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
    )

    # 2) model
    pretrained_path = args.pretrained if args.pretrained.lower() not in ("", "none", "null", "false") else None
    model = build_model(
        model_depth=args.depth,
        num_classes=args.num_classes,
        pretrained_path=pretrained_path,
        freeze_until=args.freeze_until,
        device=args.device,
    )

    # 3) one batch forward/backward
    criterion = nn.CrossEntropyLoss()
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if (not n.startswith("fc.")) and p.requires_grad]
    optimizer = optim.Adam(
        [{"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
         {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4}]
    )

    frames, targets = next(iter(train_loader))
    print(f"[DBG] Primo batch train: shape={tuple(frames.shape)} targets={targets.tolist()}")
    frames = frames.to(device).permute(0, 2, 1, 3, 4).contiguous()
    targets = targets.to(device)

    model.train()
    logits = model(frames)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    print(f"[OK] Step train di prova completato. Loss={loss.item():.4f}")

    # 4) valutazione completa su validation + salvataggi
    val_stats = eval_full(model, val_loader, device, class_names=("NORMAL", "ABNORMAL"))
    print(f"[VAL] loss={val_stats['loss']:.4f}  acc={val_stats['acc']*100:.2f}%")
    print(f"[VAL] CM:\n{val_stats['cm']}")

    # salva metriche raw
    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": float(val_stats["loss"]),
                "acc": float(val_stats["acc"]),
                "cm": val_stats["cm"].tolist(),
            },
            f, indent=2
        )

    # confusion matrix plots
    plot_cm(val_stats["cm"], ("NORMAL", "ABNORMAL"), run_dir / "confusion_matrix.png", normalize=False)
    plot_cm(val_stats["cm"], ("NORMAL", "ABNORMAL"), run_dir / "confusion_matrix_norm.png", normalize=True)

    print("[DONE] Debug completato. Artefatti salvati.")


if __name__ == "__main__":
    main()

# train.py
import argparse
import os
from pathlib import Path
import time
import json
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from DataHandler.dataloader import make_dataloaders

from model import build_model


# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def softmax_probs(logits):
    return F.softmax(logits, dim=1)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows, header, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)


def plot_confusion_matrix(cm, class_names, out_path: Path, normalize=False, title="Confusion Matrix"):
    import numpy as np
    cm = cm.cpu().numpy() if torch.is_tensor(cm) else cm
    if normalize:
        with np.errstate(all='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_norm)
        fmt = ".2f"
        data = cm_plot
    else:
        fmt = "d"
        data = cm

    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(data, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = data.max() / 2.0 if data.size else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(data[i, j], fmt),
                     ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_curves(history, out_path: Path):
    # history: dict with lists per epoch
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_path.with_name("curves_loss.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path.with_name("curves_acc.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.plot(history["val_f1_abn"], label="val_f1_abn")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("F1 (abnormal)")
    plt.tight_layout()
    plt.savefig(out_path.with_name("curves_f1_abn.png"), dpi=180)
    plt.close()


def plot_binary_curves(y_true, y_prob_abn, out_dir: Path):
    # y_true, y_prob_abn sono tensori 1D CPU (0/1), prob per classe 1
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc
    y = y_true.numpy()
    p = y_prob_abn.numpy()

    # PR curve
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=180)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=180)
    plt.close()

    # salva punti crudi
    save_csv(zip(rec, prec), ["recall", "precision"], out_dir / "pr_points.csv")
    save_csv(zip(fpr, tpr), ["fpr", "tpr"], out_dir / "roc_points.csv")
    return {"ap": float(ap), "auc": float(roc_auc)}


@torch.no_grad()
def evaluate(model, loader, device, desc="VAL", class_names=("NORMAL", "ABNORMAL")):
    model.eval()
    CE = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct = 0

    all_preds = []
    all_targets = []
    all_prob_abn = []

    for frames, targets in loader:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().to(device)  # B,C,T,H,W
        targets = targets.to(device)

        logits = model(frames)
        loss = CE(logits, targets)

        total_loss += loss.item()
        total += targets.size(0)
        probs = softmax_probs(logits)
        preds = probs.argmax(dim=1)
        correct += (preds == targets).sum().item()

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_prob_abn.append(probs[:, 1].detach().cpu())  # prob della classe 1 (ABNORMAL)

    all_preds = torch.cat(all_preds) if all_preds else torch.tensor([])
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([])
    all_prob_abn = torch.cat(all_prob_abn) if all_prob_abn else torch.tensor([])

    cm = torch.zeros(2, 2, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        cm[t.long(), p.long()] += 1

    TP = cm[1, 1].item()
    FP = cm[0, 1].item()
    FN = cm[1, 0].item()
    TN = cm[0, 0].item()

    eps = 1e-8
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)

    print(f"\n[{desc}] loss: {avg_loss:.4f} | acc: {acc*100:.2f}% | ABN-P: {prec:.3f}  ABN-R: {rec:.3f}  ABN-F1: {f1:.3f}")
    print(f"[{desc}] Confusion Matrix (rows=true, cols=pred):\n{cm}")

    return {
        "loss": avg_loss,
        "acc": acc,
        "prec_abn": prec,
        "rec_abn": rec,
        "f1_abn": f1,
        "cm": cm,
        "y_true": all_targets,        # per curve PR/ROC
        "y_prob_abn": all_prob_abn,   # per curve PR/ROC
    }


def train_one_epoch(model, loader, device, optimizer, scaler=None, class_weights=None):
    model.train()
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for frames, targets in loader:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(frames)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(1, total)
    acc = running_correct / max(1, total)
    return {"loss": avg_loss, "acc": acc}


def save_epoch_artifacts(
    epoch, epoch_dir: Path, model, optimizer, scheduler, args, train_stats, val_stats, run_history, class_names
):
    # 1) checkpoint (model+optim+scheduler+config+stats)
    ckpt = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": vars(args),
        "train_stats": train_stats,
        "val_stats": {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in val_stats.items()},
    }
    torch.save(ckpt, epoch_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    # 2) metriche in JSON + CSV
    to_json = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_acc": train_stats["acc"],
        "val_loss": val_stats["loss"],
        "val_acc": val_stats["acc"],
        "val_prec_abn": val_stats["prec_abn"],
        "val_rec_abn": val_stats["rec_abn"],
        "val_f1_abn": val_stats["f1_abn"],
    }
    save_json(to_json, epoch_dir / "metrics.json")
    save_csv(
        [[epoch, train_stats["loss"], train_stats["acc"], val_stats["loss"], val_stats["acc"], val_stats["prec_abn"], val_stats["rec_abn"], val_stats["f1_abn"]]],
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_prec_abn", "val_rec_abn", "val_f1_abn"],
        epoch_dir / "metrics.csv",
    )

    # 3) confusion matrix (png + csv)
    cm = val_stats["cm"]
    plot_confusion_matrix(cm, class_names, epoch_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, epoch_dir / "confusion_matrix_norm.png", normalize=True)
    # raw cm
    cm_list = cm.tolist() if torch.is_tensor(cm) else cm
    save_json({"labels": list(class_names), "cm": cm_list}, epoch_dir / "confusion_matrix.json")

    # 4) PR/ROC (png + csv) se disponibili
    if val_stats["y_true"].numel() > 0:
        curves = plot_binary_curves(val_stats["y_true"], val_stats["y_prob_abn"], epoch_dir)
        save_json(curves, epoch_dir / "curves_summary.json")

    # 5) curve globali aggiornate (loss/acc/f1) nella root del run
    plot_curves(run_history, epoch_dir.parent / "curves_placeholder.png")


def main():
    parser = argparse.ArgumentParser("Train 3D ResNet (walk vs run)")
    # dati
    parser.add_argument("--root", type=str, default="Dataset_jpg")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temporal-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-balance", action="store_true")
    # modello
    parser.add_argument("--depth", type=int, default=50, choices=[10, 18, 34, 50, 101, 152, 200])
    parser.add_argument("--freeze-until", type=str, default="layer3")
    parser.add_argument("--pretrained", type=str, default="PreTrained_Models/r3d50_K_200ep.pth")
    parser.add_argument("--num-classes", type=int, default=2)
    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    # output/checkpointing
    parser.add_argument("--outdir", type=str, default="FineTuned_Models")
    parser.add_argument("--run-name", type=str, default=None, help="Nome run (default: auto con timestamp)")
    parser.add_argument("--save-every", type=int, default=1, help="Salva artefatti ogni N epoche")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # classi (fissa NORMAL=0, ABNORMAL=1 nella tua dataset class per coerenza)
    class_names = ("NORMAL", "ABNORMAL")

    # --- cartelle output ---
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(args.outdir) / run_name)

    # ----------------- Data -----------------
    train_loader, val_loader, _ = make_dataloaders(
        root_path=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_size=args.temporal_size,
        image_size=args.image_size,
        balance=(not args.no_balance),
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,  # test esterno
    )

    # ----------------- Model ----------------
    model = build_model(
        model_depth=args.depth,
        num_classes=args.num_classes,
        pretrained_path=(args.pretrained if args.pretrained.lower() not in ("", "none", "null", "false") else None),
        freeze_until=(None if args.freeze_until in (None, "None", "none") else args.freeze_until),
        device=args.device,
    )

    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if (not n.startswith("fc.")) and p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr, "weight_decay": args.wd})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head, "weight_decay": args.wd})

    optimizer = optim.Adam(param_groups)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # history per curve
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1_abn": []}

    best_f1 = -1.0
    best_path = run_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, device, optimizer, scaler, class_weights=None)
        lr_scheduler.step()
        dt = time.time() - t0

        print(f"\n[EPOCH {epoch:02d}/{args.epochs}] "
              f"train_loss={train_stats['loss']:.4f}  train_acc={train_stats['acc']*100:.2f}%  "
              f"({dt:.1f}s)  lr={[g['lr'] for g in optimizer.param_groups]}")

        val_stats = evaluate(model, val_loader, device, desc="VAL", class_names=class_names)

        # aggiorna history e salva curve globali
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        history["val_f1_abn"].append(val_stats["f1_abn"])
        save_json(history, run_dir / "history.json")
        plot_curves(history, run_dir / "curves_placeholder.png")

        # salva best su F1 abnormal
        if val_stats["f1_abn"] > best_f1:
            best_f1 = val_stats["f1_abn"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_f1_abn": best_f1,
                    "config": vars(args),
                },
                best_path,
            )
            print(f"[SAVE] Nuovo best F1_ABN={best_f1:.3f} salvato in: {best_path}")

        # salva ogni N epoche (artefatti completi)
        if (epoch % max(1, args.save_every)) == 0:
            epoch_dir = ensure_dir(run_dir / f"epoch_{epoch:03d}")
            save_epoch_artifacts(
                epoch, epoch_dir, model, optimizer, lr_scheduler, args, train_stats, val_stats, history, class_names
            )

    print("\nTraining terminato. Esegui la valutazione su un set di TEST esterno separato.")
    print(f"Cartella run: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

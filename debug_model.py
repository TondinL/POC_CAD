# debug_model.py
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import build_model, get_fine_tuning_parameters


def human_count(n):
    return f"{n:,}".replace(",", ".")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def list_some_params(model, only_trainable=True, limit=15):
    items = []
    for name, p in model.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        if not only_trainable and p.requires_grad:
            continue
        items.append(name)
        if len(items) >= limit:
            break
    return items


def make_dummy(batch, channels, frames, height, width, device):
    return torch.randn(batch, channels, frames, height, width, device=device)


def main():
    parser = argparse.ArgumentParser("Debug ResNet3D custom")
    parser.add_argument("--depth", type=int, default=50, help="Profondità modello (10/18/34/50/101/152/200)")
    parser.add_argument("--num-classes", type=int, default=2, help="Classi finali (es. 2)")
    parser.add_argument("--pretrained", type=str, default="PreTrained_Models/r3d50_K_200ep.pth",
                        help="Percorso checkpoint (.pth) o vuoto per none")
    parser.add_argument("--freeze-until", type=str, default="layer3",
                        help="layer1/layer2/layer3/layer4 oppure None per non congelare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR per il test backward")
    parser.add_argument("--show-all-trainable", action="store_true", help="Stampa tutti i parametri allenabili")
    args = parser.parse_args()

    print(f"\n[INFO] Device: {args.device.upper()}")
    ckpt_path = args.pretrained if args.pretrained and args.pretrained.lower() not in ("none", "null", "false") else None
    if ckpt_path:
        p = Path(ckpt_path)
        print(f"[INFO] Checkpoint: {p}  (esiste: {p.exists()})")

    # --- build ---
    t0 = time.time()
    model = build_model(
        model_depth=args.depth,
        num_classes=args.num_classes,
        pretrained_path=ckpt_path,
        freeze_until=None if args.freeze_until in (None, "None", "none") else args.freeze_until,
        device=args.device,
    )
    t1 = time.time()
    print(f"[OK] Modello costruito in {t1 - t0:.2f}s")

    # --- conti parametri ---
    total, trainable, frozen = count_params(model)
    print(f"[STATS] Parametri totali  : {human_count(total)}")
    print(f"[STATS] Parametri train   : {human_count(trainable)}")
    print(f"[STATS] Parametri frozen  : {human_count(frozen)}")

    some_train = list_some_params(model, only_trainable=True, limit=15)
    some_frozen = list_some_params(model, only_trainable=False, limit=15)
    print("\n[CHK] Alcuni parametri ALLENABILI:")
    for n in some_train:
        print("  •", n)
    print("\n[CHK] Alcuni parametri CONGELATI:")
    for n in some_frozen:
        print("  •", n)

    if args.show_all_trainable:
        print("\n[ALL TRAINABLE] Parametri allenabili (tutti):")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print("  •", n)

    # --- forward eval su input dummy ---
    dummy = make_dummy(args.batch, args.channels, args.frames, args.height, args.width, args.device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"\n[FORWARD-EVAL] Output shape: {tuple(out.shape)}  (atteso: ({args.batch}, {args.num_classes}))")

    # --- test loss + backward solo per verificare gradienti sui layer sbloccati ---
    model.train()
    criterion = nn.CrossEntropyLoss()
    # param groups di fine-tuning: esempio (fc con LR più alto)
    ft_params = get_fine_tuning_parameters(model, ft_begin_module="layer3")
    if not ft_params:  # fallback: tutti i parametri
        ft_params = [{"params": p} for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(ft_params, lr=args.lr)

    # target fittizio
    y = torch.randint(low=0, high=args.num_classes, size=(args.batch,), device=args.device)
    optimizer.zero_grad()
    out2 = model(dummy)
    loss = criterion(out2, y)
    loss.backward()

    # --- verifica gradienti: i frozen devono avere grad=None, i trainable grad!=None ---
    missing_grad = []
    wrong_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing_grad.append(name)
        if (not p.requires_grad) and (p.grad is not None):
            wrong_grad.append(name)

    print(f"\n[BACKWARD] Loss: {loss.item():.4f}")
    if missing_grad:
        print("[WARN] Parametri ALLENABILI senza gradiente (controlla freeze/graph):")
        for n in missing_grad[:20]:
            print("  •", n)
        if len(missing_grad) > 20:
            print(f"  … (+{len(missing_grad)-20} altri)")
    else:
        print("[OK] Tutti i parametri allenabili hanno gradiente.")

    if wrong_grad:
        print("[ERR] Parametri CONGELATI con gradiente (non dovrebbero!):")
        for n in wrong_grad[:20]:
            print("  •", n)
        if len(wrong_grad) > 20:
            print(f"  … (+{len(wrong_grad)-20} altri)")
    else:
        print("[OK] Nessun grad su parametri congelati.")

    # un singolo step per assicurare che l'optimizer funzioni
    optimizer.step()
    print("[OK] Optimizer step eseguito.")

    print("\n[DEBUG COMPLETATO] Se output shape, freeze e gradienti sono ok, la pipeline è corretta.")


if __name__ == "__main__":
    sys.exit(main())

# models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Se questo file è nella stessa cartella di resnet.py
from PreTrained_Models import resnet


def build_model(
    model_depth: int = 50,
    num_classes: int = 2,
    # parametri architetturali (devono combaciare con i pesi pretrain!)
    n_input_channels: int = 3,
    conv1_t_size: int = 7,
    conv1_t_stride: int = 1,
    no_max_pool: bool = False,
    shortcut_type: str = "B",
    widen_factor: float = 1.0,
    # fine-tuning
    pretrained_path: Optional[str] = "PreTrained_Models/r3d50_K_200ep.pth",
    freeze_until: Optional[str] = "layer3",  # "layer1"/"layer2"/"layer3"/"layer4" oppure None
    device: str = "cpu",
) -> nn.Module:
    """
    Crea una ResNet3D (dal tuo resnet.py), carica eventuali pesi pre-addestrati,
    congela i layer richiesti e sostituisce la testa finale per num_classes.
    """
    # 1) costruttore backbone (n_classes temporaneo = classi del pretrain, p.es. 400)
    #    NB: se i tuoi pesi sono kinetics-400, lascia 400; se sono 700, cambia qui.
    model = resnet.generate_model(
        model_depth=model_depth,
        n_classes=400,  # <-- adatta se il checkpoint è su 700 classi
        n_input_channels=n_input_channels,
        conv1_t_size=conv1_t_size,
        conv1_t_stride=conv1_t_stride,
        no_max_pool=no_max_pool,
        shortcut_type=shortcut_type,
        widen_factor=widen_factor,
    )

    # 2) carica pesi (se forniti)
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)

        # rimuovi prefisso "module."
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # ignora la fc del checkpoint (verrà rimpiazzata per num_classes)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}

        res = model.load_state_dict(state, strict=False)
        print(f"[INFO] Pesi caricati da: {pretrained_path}")
        if res.missing_keys:
            print(f"[WARN] Missing keys: {res.missing_keys}")
        if res.unexpected_keys:
            print(f"[WARN] Unexpected keys: {res.unexpected_keys}")
    else:
        print("[INFO] Nessun pretrained_path: inizializzazione casuale.")

    # 3) congela fino a freeze_until
    if freeze_until is not None:
        _freeze_backbone(model, freeze_until)

    # 4) sostituisci la testa
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    print(f"[INFO] freeze_until = {freeze_until}")
    print(f"[INFO] num_classes  = {num_classes}")
    return model.to(device)


def _freeze_backbone(model: nn.Module, freeze_until: str) -> None:
    """
    Congela parametri finché non si raggiunge il modulo indicato.
    Valori attesi: 'layer1'/'layer2'/'layer3'/'layer4'
    """
    valid = {"layer1", "layer2", "layer3", "layer4"}
    if freeze_until not in valid:
        print(f"[WARN] freeze_until='{freeze_until}' non è in {valid}. Nessun freeze applicato.")
        return

    freeze = True
    for name, p in model.named_parameters():
        if freeze:
            p.requires_grad = False
        # quando troviamo il modulo target, da qui in poi sblocchiamo
        if freeze_until in name:
            freeze = False

    # stampa diagnostica
    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"[INFO] Parametri frozen: {frozen}, trainable: {trainable}")


# ---------- utilità opzionali ----------

def get_fine_tuning_parameters(
    model: nn.Module,
    ft_begin_module: Optional[str],
) -> list[dict]:
    """
    Ritorna param groups a partire da un certo modulo (incluso).
    Es: ft_begin_module='layer3' --> layer3/layer4 + fc allenabili.
    Se ft_begin_module è None, ritorna tutti i parametri.
    """
    if not ft_begin_module:
        return [{"params": p} for p in model.parameters()]

    params = []
    add_flag = False
    for k, v in model.named_parameters():
        # normalizza nome modulo top-level (conv1, layer1, layer2, layer3, layer4, fc)
        top = k.split(".")[0]
        if top == ft_begin_module:
            add_flag = True
        if add_flag:
            params.append({"params": v})
    return params


def make_data_parallel(
    model: nn.Module,
    is_distributed: bool,
    device: torch.device,
) -> nn.Module:
    """
    Wrapper per DP/DDP.
    - is_distributed=True => DistributedDataParallel
    - altrimenti, se CUDA => DataParallel
    """
    if is_distributed:
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        model = nn.DataParallel(model).cuda()
    else:
        model.to(device)
    return model


# ---------- mini test locale ----------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(
        model_depth=50,
        num_classes=2,
        pretrained_path=None,     # metti il path ai tuoi pesi per provare il load
        freeze_until="layer3",
        device=dev,
    )
    x = torch.randn(2, 3, 16, 112, 112).to(dev)  # (B, C, T, H, W)
    net.eval()
    with torch.no_grad():
        y = net(x)
    print("[CHK] output shape:", y.shape)  # atteso: (2, 2)

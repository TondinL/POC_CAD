# model_finetune_3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_50


def build_finetune_model_3d(
    num_classes=2,
    pretrained_path="PreTrained_Models/r3d50_K_200ep.pth",
    freeze_until="layer3",
    device="cpu"
):
    """
    Crea un modello ResNet3D-50 e carica i pesi preaddestrati (es. Kinetics).

    Args:
        num_classes (int): numero di classi del tuo dataset (es. 2)
        pretrained_path (str): percorso al file .pth dei pesi preaddestrati
        freeze_until (str): congela i layer fino a questo nome (es. 'layer3')
        device (str): 'cpu' o 'cuda'

    Returns:
        torch.nn.Module pronto per il fine-tuning
    """
    # ---  Crea il modello base ---
    model = r3d_50(weights=None)  # non carica pesi ImageNet

    # ---  Carica pesi pre-addestrati ---
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device)
        # alcuni modelli salvano sotto 'state_dict', altri no
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        # rimuovi eventuale prefisso 'module.' dai nomi dei layer
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f" Pesi pre-addestrati caricati da {pretrained_path}")
    else:
        print(" Nessun file di pesi fornito — modello inizializzato casualmente")

    # ---  Congela parte dei layer ---
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False

    # ---  Sostituisci la testa finale ---
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    print(f" Layer congelati fino a: {freeze_until}")
    print(f" Uscita finale: {num_classes} classi")

    return model.to(device)

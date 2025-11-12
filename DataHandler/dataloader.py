import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

# Import RELATIVI perché questo file sta dentro DataHandler/
from .dataset_video import VideoDataset
from .temporal_transforms import TemporalRandomCropStrict
from .balanced_augmentation import BalancedAugmentation


def make_dataloaders(
    root_path,
    batch_size=4,
    num_workers=4,
    train_split=0.8,
    val_split=0.2,
    test_split=0,
    temporal_size=8,
    image_size=224,
    balance=True
):
    """
    Crea DataLoader per training, validation e test da dataset di frame video.

    root_path: cartella principale con sottocartelle NORMAL/ABNORMAL
    balance: se True, usa WeightedRandomSampler per bilanciare le classi
    """

    # --- Trasformazioni ---
    temporal_transform = TemporalRandomCropStrict(size=temporal_size)
    spatial_transform = BalancedAugmentation(image_size=image_size)

    # --- Dataset completo ---
    full_dataset = VideoDataset(
        root_path=root_path,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform
    )

    # --- Split train / val / test ---
    total_len = len(full_dataset)
    n_train = int(train_split * total_len)
    n_val = int(val_split * total_len)
    n_test = total_len - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # --- Bilanciamento con WeightedRandomSampler ---
    if balance:
        # conta le classi nel train set
        labels = [full_dataset.samples[i][1] for i in train_set.indices]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1. / class_counts.float()
        sample_weights = [class_weights[l] for l in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # --- Dataloader ---
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n Split dataset: {n_train} train, {n_val} val, {n_test} test")
    print(f"  Bilanciamento: {'attivo' if balance else 'disattivato'}")

    return train_loader, val_loader, test_loader

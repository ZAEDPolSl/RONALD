from monai.data import DataLoader, CacheDataset
from .image_loader import load_image_with_instance


def get_dataloaders(
    train_files,
    val_files,
    test_files,
    train_transforms,
    val_transforms,
    batch_size,
    num_workers,
    cache_rate=1.0,
):
    # Use a loader that supports multiple extensions via ImageInstance
    def image_loader(item):
        item = dict(item)
        item["image"] = load_image_with_instance(item["image"])
        if "label" in item:
            item["label"] = load_image_with_instance(item["label"])
        return item

    train_ds = CacheDataset(
        data=train_files,
        transform=lambda x: train_transforms(image_loader(x)),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=lambda x: val_transforms(image_loader(x)),
        cache_rate=1.0,
        num_workers=num_workers,
    )
    test_ds = (
        CacheDataset(
            data=test_files,
            transform=lambda x: val_transforms(image_loader(x)),
            cache_rate=1.0,
            num_workers=num_workers,
        )
        if test_files
        else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = (
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        if test_ds
        else None
    )
    return train_loader, val_loader, test_loader

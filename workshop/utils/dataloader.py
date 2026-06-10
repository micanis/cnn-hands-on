from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ImageTransform = Callable[[Image.Image], torch.Tensor]

CLASS_ALIASES = {
    "cat": 0,
    "cats": 0,
    "Cat": 0,
    "Cats": 0,
    "dog": 1,
    "dogs": 1,
    "Dog": 1,
    "Dogs": 1,
}
CLASS_NAMES = ["cat", "dog"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_SIZE_RATIOS = {
    "small": 0.2,
    "medium": 0.5,
    "large": 1.0,
}


class CDDataset(Dataset):
    """Cat/Dog image dataset used in the classification notebooks."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str | None = None,
        transform: ImageTransform | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_default_transform()
        self.samples = _collect_samples(self.data_dir, split)

        if not self.samples:
            split_text = f" split={split!r}" if split else ""
            raise FileNotFoundError(
                f"No cat/dog images were found in {self.data_dir}{split_text}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def get_default_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def get_imagenet_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_cat_dog_dataloaders(
    root_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    data_size: str = "large",
    batch_size: int = 32,
    image_size: int = 128,
    transform: ImageTransform | None = None,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders for cat/dog classification.

    The function accepts either an explicit ``data_dir`` or a course ``root_path``.
    ``data_size`` can be ``small`` (20%), ``medium`` (50%), or ``large`` (100%).
    """

    dataset_root = _resolve_dataset_root(root_path=root_path, data_dir=data_dir)
    transform = transform or get_default_transform(image_size)

    if _has_split_dirs(dataset_root):
        train_dataset = CDDataset(dataset_root, split="train", transform=transform)
        val_dataset = CDDataset(dataset_root, split="val", transform=transform)
        test_dataset = CDDataset(dataset_root, split="test", transform=transform)
    else:
        full_dataset = CDDataset(dataset_root, transform=transform)
        train_dataset, val_dataset, test_dataset = _split_dataset(full_dataset, seed)

    train_dataset = _limit_dataset(train_dataset, data_size=data_size, seed=seed)
    val_dataset = _limit_dataset(val_dataset, data_size=data_size, seed=seed)
    test_dataset = _limit_dataset(test_dataset, data_size=data_size, seed=seed)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def get_cd_dataloaders(*args, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    return get_cat_dog_dataloaders(*args, **kwargs)


def get_dc_dataloaders(*args, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    return get_cat_dog_dataloaders(*args, **kwargs)


def _resolve_dataset_root(
    root_path: str | Path | None,
    data_dir: str | Path | None,
) -> Path:
    if data_dir is not None:
        return Path(data_dir)

    if root_path is None:
        raise ValueError("Either root_path or data_dir must be provided.")

    root = Path(root_path)
    candidates = [
        root / "data" / "cat_dog",
        root / "data" / "cats_dogs",
        root / "data" / "dog_cat",
        root / "data" / "dogs_cats",
        root / "cat_dog",
        root / "cats_dogs",
        root / "dog_cat",
        root / "dogs_cats",
        root / "dataset",
        root / "data",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _collect_samples(data_dir: Path, split: str | None) -> list[tuple[Path, int]]:
    search_roots = _split_roots(data_dir, split)
    samples: list[tuple[Path, int]] = []

    for root in search_roots:
        for class_dir in _class_dirs(root):
            label = CLASS_ALIASES[class_dir.name]
            samples.extend((path, label) for path in _image_files(class_dir))

        if samples:
            continue

        for path in _image_files(root):
            label = _label_from_filename(path)
            if label is not None:
                samples.append((path, label))

    samples.sort(key=lambda item: str(item[0]))
    return samples


def _split_roots(data_dir: Path, split: str | None) -> list[Path]:
    if split is None:
        return [data_dir]

    aliases = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation"],
        "test": ["test", "testing"],
    }[split]

    roots = [data_dir / alias for alias in aliases if (data_dir / alias).exists()]
    return roots or [data_dir]


def _has_split_dirs(data_dir: Path) -> bool:
    return all(_split_roots(data_dir, split)[0] != data_dir for split in ["train", "val", "test"])


def _class_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name in CLASS_ALIASES
    ]


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _label_from_filename(path: Path) -> int | None:
    name = path.stem.lower()
    if name.startswith("cat"):
        return 0
    if name.startswith("dog"):
        return 1
    return None


def _limit_dataset(
    dataset: Dataset,
    data_size: str,
    seed: int,
) -> Dataset:
    if data_size not in DATA_SIZE_RATIOS:
        choices = ", ".join(DATA_SIZE_RATIOS)
        raise ValueError(f"data_size must be one of: {choices}")

    ratio = DATA_SIZE_RATIOS[data_size]
    if ratio >= 1.0:
        return dataset

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    keep_count = max(1, int(len(indices) * ratio))
    return Subset(dataset, sorted(indices[:keep_count]))


def _split_dataset(dataset: Dataset, seed: int) -> tuple[Subset, Subset, Subset]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    train_end = max(1, int(len(indices) * 0.7))
    val_end = max(train_end + 1, int(len(indices) * 0.85))
    val_end = min(val_end, len(indices) - 1)

    train_indices = sorted(indices[:train_end])
    val_indices = sorted(indices[train_end:val_end])
    test_indices = sorted(indices[val_end:])

    if not val_indices:
        val_indices = train_indices[-1:]
    if not test_indices:
        test_indices = val_indices[-1:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

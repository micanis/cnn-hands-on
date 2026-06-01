import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

_FRAC_MAP = {"small": 0.2, "medium": 0.5, "large": 1.0}


class CDDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.root_dir, row["filepath"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label"]


def get_cd_dataloaders(root_path, data_size="large", batch_size=32):
    frac = _FRAC_MAP.get(data_size)
    if frac is None:
        raise ValueError(
            f"Invalid data_size: {data_size!r}. Choose from {list(_FRAC_MAP)}"
        )

    data_dir = os.path.join(root_path, "data", "cats_vs_dogs")
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    def make_loader(split, shuffle):
        dataset = CDDataset(
            df[df["split"] == split].sample(frac=frac, random_state=61),
            data_dir,
            transform,
        )
        kwargs = {"num_workers": 2, "pin_memory": True} if shuffle else {}
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return (
        make_loader("train", True),
        make_loader("val", False),
        make_loader("test", False),
    )

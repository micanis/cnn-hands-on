import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CatDogDataset(Dataset):
    """
    犬猫データセット用
    """
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.loc[idx, "filepath"]
        img_path = os.path.join(self.root_dir, img_rel_path)

        image = Image.open(img_path).convert("RGB")

        label = self.df.loc[idx, "label"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_cat_dog_dataloaders(root_path, data_size="large", batch_size=32):
    """
    指定サイズのデータローダーを返す

    data_size: "small"(20%), "medimum"(50%), "large"(100%)
    """
    data_dir = os.path.join(root_path, "data", "cats_vs_dogs")
    csv_path = os.path.join(data_dir, "labels.csv")

    df = pd.read_csv(csv_path)

    if data_size == "small":
        frac = 0.2
    elif data_size == "medium":
        frac = 0.5
    elif data_size == "large":
        frac = 1.0
    else:
        print("data_size NameError")
    
    train_df = df[df["split"] == "train"].sample(frac=frac, random_state=61)
    val_df = df[df["split"] == "val"].sample(frac=frac, random_state=61)
    test_df = df[df["split"] == "test"].sample(frac=frac, random_state=61)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = CatDogDataset(train_df, data_dir, transform=transform)
    val_dataset = CatDogDataset(val_df, data_dir, transform=transform)
    test_dataset = CatDogDataset(test_df, data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
from src.transforms import get_transforms
import torch


class create_dataset(Dataset):
    """
    Custom dataset class for creating a PyTorch dataset from a DataFrame.
    """

    def __init__(self, items, trans=None, labelled=True):
        self.items = items
        self.trans = trans
        self.labelled = labelled

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]

        if self.labelled:
            file_path = item["file_path"]
            label = item["label"]
        else:
            file_path = item
            label = None

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.trans:
            img = self.trans(image=img)["image"]

        if label is None:
            return img
        else:
            return img, torch.tensor(label, dtype=torch.float32)


def create_data_loaders(
    df, batch_size=32, img_size=(50, 50), val_split=0.2, labelled=True
):
    train_transform, val_transform = get_transforms(img_size)

    if labelled:
        train_ds, val_ds = train_test_split(
            df, test_size=val_split, random_state=17, shuffle=True
        )
        train_dict = train_ds.to_dict(orient="records")
        val_dict = val_ds.to_dict(orient="records")

        train_dataset = create_dataset(train_dict, trans=train_transform)
        val_dataset = create_dataset(val_dict, trans=val_transform)

        train_load = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )
        val_load = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

        return train_load, val_load
    else:
        test_dataset = create_dataset(df, trans=val_transform, labelled=False)
        test_load = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

        return test_load

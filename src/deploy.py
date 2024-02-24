import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import get_augmentation, HistoCRCDataset
from model import HistoClassifier
from utils import validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = Path(
    "/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/models/efficientnet-b0_binary=False_2024-02-24T11:48:56.609"
)
model = HistoClassifier.from_pretrained(model_path)
model.to(device)


data_dir = Path(
    "/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/"
)
valid_dir = data_dir / "CRC-VAL-HE-7K"
batch_size = 64

img_size = (model.config.inp_height, model.config.inp_width)
mean, std = model.config.mean, model.config.std
valid_aug = get_augmentation(img_size, mean, std, validation=True)

# initialize datasets and dataloaders
valid_ds = HistoCRCDataset(valid_dir, augmentation=valid_aug, reduce_to_binary=model.config.n_classes == 2)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
)
valid_ds.describe()


X, y =  next(iter(valid_dl))
X, y = X.to(device), y.numpy() 
pred = model.predict_category(X)
print(pred, y)
pred = model.predict_category(X, label=True)
print(pred, y)

report = validate(model, valid_dl)
print(report)

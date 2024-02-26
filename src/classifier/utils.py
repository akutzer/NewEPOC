from typing import Iterable

import torch
from torch import nn
from sklearn.metrics import classification_report
from tqdm import tqdm


def validate(model: nn.Module, valid_dl: Iterable) -> str:
    device = next(model.parameters()).device
    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for X, y in tqdm(valid_dl):
            X, y = X.to(device), y.to(device)
            out = model(X)
            y_preds += torch.argmax(out, dim=-1).tolist()
            y_trues += y.tolist()

    report = classification_report(
        y_trues, y_preds, target_names=valid_dl.dataset.categories, digits=4
    )
    model.train()
    return report


def validate_binarized(model: nn.Module, valid_dl: Iterable) -> str:
    device = next(model.parameters()).device
    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for X, y in valid_dl:
            X, y = X.to(device), y.to(device)
            out = torch.softmax(model(X), dim=-1)
            y_preds += (out[:, 7:].sum(dim=-1) > out[:, :7].sum(dim=-1)).int().tolist()
            y_trues += (y >= 7).int().tolist()

    report = classification_report(
        y_trues, y_preds, target_names=["NORM", "TUM"], digits=4
    )
    model.train()
    return report

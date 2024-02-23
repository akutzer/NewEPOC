
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import os
from pathlib import Path
# from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, BinaryAccuracy
from sklearn.metrics import classification_report

class HistoClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dim, n_classes))
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

    def forward(self, *args, **kwargs):
        out = self.backbone(*args, **kwargs).pooler_output
        out = self.head(out)
        return out


def load_model(model_name, n_classes=None):
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "hidden_dim"):
            hidden_dim = config.hidden_dim
    elif hasattr(config, "hidden_size"):
        hidden_dim = config.hidden_size
    else:
        raise AttributeError
    
    is_local = os.path.isdir(model_name)
    if is_local:
        model_dir = Path(model_name)
        backbone = AutoModel.from_config(config)
        state_dict = torch.load(model_dir/"pytorch_model.bin")
        n_classes = state_dict["head.1.bias"].shape[0]
        model = HistoClassifier(backbone, hidden_dim, n_classes)
        
        model.load_state_dict(state_dict)
    else:
        assert isinstance(n_classes, int)
        backbone = AutoModel.from_pretrained(model_name)
        model = HistoClassifier(backbone, hidden_dim, n_classes)

    return model, config


def validate(model, valid_dl):
    device = next(model.parameters()).device

    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for X, y in valid_dl:
            X, y = X.to(device), y.to(device)
            out = model(X)
            y_preds += torch.argmax(out, dim=-1).tolist()
            y_trues += y.tolist()
                
    report = classification_report(y_trues, y_preds, target_names=valid_dl.dataset.categories, digits=4)
    return report

def validate_multi_class(model, valid_dl):
    device = next(model.parameters()).device
    
    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for X, y in valid_dl:
            X, y = X.to(device), y.to(device)
            out = torch.softmax(model(X), dim=-1)
            y_preds += (out[:, 7:].sum(dim=-1) > out[:, :7].sum(dim=-1)).int().tolist()
            y_trues += (y >= 7).int().tolist()
                
    report = classification_report(y_trues, y_preds, target_names=["NORM", "TUM"], digits=4)
    return report

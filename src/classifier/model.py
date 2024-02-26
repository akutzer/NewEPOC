from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig


class HistoClassifierConfig(PretrainedConfig):
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        n_classes: Optional[int] = None,
        inp_height: Optional[int] = None,
        inp_width: Optional[int] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        backbone: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.categories = categories
        self.n_classes = n_classes
        self.inp_height = inp_height
        self.inp_width = inp_width
        self.mean = mean
        self.std = std
        self.backbone = backbone
        self.hidden_dim = hidden_dim

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.update(
            {
                "categories": self.categories,
                "n_classes": self.n_classes,
                "inp_height": self.inp_height,
                "inp_width": self.inp_width,
                "mean": self.mean,
                "std": self.std,
                "backbone": self.backbone,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config_dict


class HistoClassifier(nn.Module):
    def __init__(self, backbone: str, hidden_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dim, n_classes))
        self.config = None

    def forward(self, x) -> torch.Tensor:
        out = self.backbone(x).pooler_output
        out = self.head(out)
        return out

    def predict(self, x) -> np.ndarray:
        cache_mode = self.training
        self.train(False)
        if unsqueeze := (x.dim() == 3):
            x = x.unsqueeze(0)

        with torch.inference_mode():
            pred = self.head(self.backbone(x).pooler_output)
            probs = torch.softmax(pred, dim=-1).cpu()
            if unsqueeze:
                probs = probs[0]
        probs, indices = torch.sort(probs, descending=True)
        probs, indices = probs.numpy(), indices.numpy()
        labels = np.take(self.config.categories, indices)

        out = np.empty(
            probs.shape,
            dtype=np.dtype(
                [
                    ("label", labels.dtype),
                    ("id", indices.dtype),
                    ("probability", probs.dtype),
                ]
            ),
        )
        out["probability"] = probs
        out["id"] = indices
        out["label"] = labels

        self.train(cache_mode)
        return out

    @classmethod
    def from_backbone(cls, backbone_name: str, categories: List[str]):
        config = AutoConfig.from_pretrained(backbone_name)
        if hasattr(config, "hidden_dim"):
            hidden_dim = config.hidden_dim
        elif hasattr(config, "hidden_size"):
            hidden_dim = config.hidden_size
        else:
            raise AttributeError
        config = HistoClassifierConfig(**config.to_dict())
        config.update(
            {
                "hidden_dim": hidden_dim,
                "n_classes": len(categories),
                "id2label": {i: cat for i, cat in enumerate(categories)},
                "label2id": {cat: i for i, cat in enumerate(categories)},
            }
        )

        backbone = AutoModel.from_pretrained(backbone_name)
        model = cls(backbone, config.hidden_dim, config.n_classes)
        model.config = config
        return model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, n_classes: Optional[int] = None
    ):
        model_dir = Path(pretrained_model_path)
        assert model_dir.is_dir()

        config = HistoClassifierConfig.from_pretrained(model_dir)
        backbone = AutoModel.from_pretrained(config.backbone)
        model = cls(backbone, config.hidden_dim, config.n_classes)
        state_dict = torch.load(model_dir / "model.pt")
        model.load_state_dict(state_dict)
        model.config = config

        return model

    def save_pretrained(self, path: str):
        path = Path(path)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "model.pt")

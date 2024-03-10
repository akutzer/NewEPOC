from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, PretrainedConfig
from tqdm import tqdm

from tissuemap.classifier.data import get_augmentation
from tissuemap.features.extractor.feature_extractors import load_ctranspath



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
        is_ctranspath: bool = False,
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
        self.is_ctranspath = is_ctranspath

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
                "is_ctranspath": self.is_ctranspath,
            }
        )
        return config_dict


class HistoClassifier(nn.Module):
    def __init__(self, backbone: str, hidden_dim: int, n_classes: int, is_ctranspath: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dim, n_classes))
        self.config = None
        self.device = None
        self.dtype = next(self.parameters()).dtype
        self.is_ctranspath = is_ctranspath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if not self.is_ctranspath:
            out = out.pooler_output
        out = self.head(out)
        return out

    def predict(self, x) -> np.ndarray:
        cache_mode = self.training
        self.train(False)
        if unsqueeze := (x.dim() == 3):
            x = x.unsqueeze(0)

        with torch.inference_mode():
            pred = self.head(self.backbone(x) if self.is_ctranspath else self.backbone(x).pooler_output)
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
    
    def predict_patches(
            self, patches: np.ndarray, cores: int = 8, batch_size: int = 64
    ) -> np.ndarray:
        img_size = (self.config.inp_height, self.config.inp_width)
        mean, std = self.config.mean, self.config.std
        transform = get_augmentation(img_size, mean, std, validation=True)

        dataset = SlideTileDataset(patches, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cores,
            drop_last=False,
            pin_memory=self.device != torch.device("cpu"),
        )

        features = []
        with torch.inference_mode():
            for patches_batch in tqdm(dataloader, leave=False):
                patches_batch = patches_batch.to(dtype=self.dtype, device=self.device)
                features_batch = self.predict(patches_batch)
                features.append(features_batch)

        features = np.concatenate(features, axis=0)
        return features

    @classmethod
    def from_backbone(cls, backbone_name: str, categories: List[str], device: str = "cpu"):
        is_ctranspath = "ctranspath" in backbone_name
        if is_ctranspath:
            # TODO: refactor
            backbone = load_ctranspath(backbone_name, device=device)
            # backbone = FeatureExtractor.from_checkpoint(backbone_name, device=device).model
            config = HistoClassifierConfig(
                categories = categories,
                n_classes = len(categories),
                inp_height = 224,
                inp_width = 224,
                mean= [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                backbone = "ctranspath",
                hidden_dim = 768,
                is_ctranspath=True
            )
            config.update({
                "id2label": {i: cat for i, cat in enumerate(categories)},
                "label2id": {cat: i for i, cat in enumerate(categories)},
            })
        else:
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
        model = cls(backbone, config.hidden_dim, config.n_classes, is_ctranspath=is_ctranspath)
        model.config = config
        model.to(device)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str = "cpu"):
        model_dir = Path(pretrained_model_path)
        assert model_dir.is_dir()

        config = HistoClassifierConfig.from_pretrained(model_dir)
        if config.is_ctranspath:
            backbone = load_ctranspath(device=device)
        else:
            backbone = AutoModel.from_pretrained(config.backbone)
        model = cls(backbone, config.hidden_dim, config.n_classes, config.is_ctranspath)
        state_dict = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.config = config
        model.to(device)
        return model

    def save_pretrained(self, path: str):
        path = Path(path)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "model.pt")

    def to(self, device: str, dtype=None):
        self.device = torch.device(device)
        return super().to(self.device, dtype)


class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None) -> None:
        self.tiles = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i) -> torch.Tensor:
        image = self.tiles[i]
        if self.transform:
            image = self.transform(image)

        return image
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def get_imgs(directory: Path) -> List[str]:
    endings = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]
    img_paths = [
        str(path) for ending in endings for path in directory.glob(f"**/*.{ending}")
    ]
    return img_paths


def get_augmentation(
    img_size: int = 224,
    mean: Sequence[float] = [0.485, 0.456, 0.406],
    std: Sequence[float] = [0.229, 0.224, 0.225],
    validation: bool = False,
) -> v2.Transform:
    transform = [v2.ToDtype(torch.float32, scale=True)]
    if validation:
        transform.append(v2.Resize(img_size))
    else:
        transform.extend(
            [
                v2.RandomResizedCrop(size=img_size, scale=(0.5, 1)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.06, 0.06),
                ),
                v2.RandomAdjustSharpness(sharpness_factor=3, p=0.25),
                v2.RandomAutocontrast(p=0.25),
                # v2.GaussianBlur(kernel_size=(1, 7), sigma=(0.5, 2.)),
            ]
        )
    transform.append(v2.Normalize(mean, std, inplace=True))
    return v2.Compose(transform)


class HistoCRCDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        augmentation: Optional[v2.Transform] = None,
        reduce_to_binary: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.augmentation = augmentation

        cat_img_map = {
            x.stem: imgs
            for x in self.img_dir.iterdir()
            if x.is_dir() and len(imgs := get_imgs(x)) > 0
        }
        if reduce_to_binary:
            tum_cat = {"TUM", "STR"}
            cat_img_map = {
                "NORM": sum((cat_img_map[cat] for cat in set(cat_img_map.keys()) - tum_cat), []),
                "TUM": sum((cat_img_map[cat] for cat in tum_cat), []),
            }
        self.categories = list(cat_img_map.keys())
        self.n_classes = len(self.categories)

        img_paths = sum(list(cat_img_map.values()), [])
        labels = [
            self.categories.index(cat)
            for cat, paths in cat_img_map.items() for _ in paths
        ]
        categories = [cat for cat, paths in cat_img_map.items() for _ in paths]
        self.data = pd.DataFrame(
            {"path": img_paths, "label": labels, "category": categories}
        )
        self._transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.data.iloc[idx, :2]
        image = self._transform(Image.open(img_path).convert("RGB"))
        if self.augmentation:
            image = self.augmentation(image)
        return image, label

    def inv_weights(self) -> torch.Tensor:
        counts = self.data["category"].value_counts(sort=False).values
        counts = torch.from_numpy(counts)
        w = counts.sum() / counts
        w = w / w.sum()
        return w

    def describe(self):
        print(f"Categories ({self.n_classes}): ", self.categories)
        counts = self.data["category"].value_counts(sort=False)
        distr = counts / counts.sum()
        distr.rename("distribution", inplace=True)
        df = pd.concat([counts, distr], axis=1, join="inner")
        print("Distribution of categories:\n  ", str(df).replace("\n", "\n  "))
        print(f"Total: {len(self)}")
        print(f"Augmentation:", self.augmentation)


def plot_grid(dataset, N: int = 5):
    """ Plots a random slide and N² - 1 augmentations of the same slide. """
    fig, axs = plt.subplots(N, N, figsize=(10, 10))
    idx = torch.randint(0, len(dataset), (1,)).item()
    
    aug, dataset.augmentation = dataset.augmentation, None
    original = dataset[idx][0]
    dataset.augmentation = aug

    axs[0, 0].imshow(original.permute(1, 2, 0).numpy())
    for k in range(1, N * N):
        i, j = k // N, k % N
        img, label = dataset[idx]
        axs[i, j].imshow(img.permute(1, 2, 0).numpy())
    plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    img_dir = "/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/CRC-VAL-HE-7K"
    aug = get_augmentation(600, mean=[0, 0, 0], std=[1, 1, 1], validation=False)
    dataset = HistoCRCDataset(img_dir, augmentation=aug, reduce_to_binary=False)
    dataset.describe()
    plot_grid(dataset, N=5)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    X, y = next(iter(train_dataloader))
    print(X.shape, y.shape)

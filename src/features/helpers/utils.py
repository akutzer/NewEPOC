from pathlib import Path
from PIL import Image
import math

from sklearn.feature_extraction.image import reconstruct_from_patches_2d



def reconstruct_from_patches(patches, img_shape):
    print("Reconstructing image from patches...")

    img_h, img_w = img_shape
    patch_h, patch_w = patches.shape[1:3]
    img = Image.new("RGB", (img_w, img_h))
    cols, rows = math.ceil(img_w / patch_w), math.ceil(img_h / patch_h)

    for k, patch in enumerate(patches):
        row, col = k // cols, k % cols
        x, y = col * patch_w, row * patch_h
        w = patch_w if col < cols - 1 else (img_w - x)
        h = patch_h if row < rows - 1 else (img_h - y)

        img.paste(
            Image.fromarray(patch[:h, :w]),
            (x, y, x + w, y + h)
        )
    return img

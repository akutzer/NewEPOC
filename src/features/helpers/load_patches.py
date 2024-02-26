import math
import numpy as np
from PIL import Image



def reconstruct_from_patches(patches, patches_coords, img_shape):
    img_h, img_w = img_shape
    patch_h, patch_w = patches.shape[1:3]
    img = Image.new("RGB", (img_w, img_h))
    for (x, y), patch in zip(patches_coords, patches):
        img.paste(
            Image.fromarray(patch[:patch_h, :patch_w]),
            (y, x, y + patch_w, x + patch_h)
        )
    return img


def extract_patches(img, patch_size, pad=True):
    patch_size = np.array(patch_size)
    if pad:
        rows, cols = np.ceil(np.array(img.shape)[:2] / patch_size).astype(int)
    else:
        rows, cols = np.array(img.shape)[:2] // patch_size
    n = rows * cols

    # num of patches x 224 x 224 x 3 for RGB patches
    patches = np.zeros((n, patch_size[0], patch_size[1], img.shape[-1]), dtype=np.uint8)
    patches_coords = np.zeros((n, 2), dtype=np.uint16)

    for i in range(rows):
        for j in range(cols):
            k = i*cols + j
            x, y = i*patch_size[0], j*patch_size[1]
            patch = img[x:x+patch_size[0], y:y+patch_size[1]]
            # zero pad on the left and bottom so all patches have the same size
            if pad and ((real_shape := np.array(patch.shape[:2])) < patch_size).any():
                padding = patch_size - real_shape
                patch = np.pad(patch, pad_width=((0, padding[0]), (0, padding[1]), (0, 0)))
            
            patches[k] = patch
            patches_coords[k] = (x, y)

    return patches, patches_coords


def reconstruct_from_patches_(patches, img_shape):
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

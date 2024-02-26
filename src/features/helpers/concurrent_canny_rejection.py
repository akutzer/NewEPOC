import time
from typing import Dict, Tuple, List, Any
from concurrent import futures
import numpy as np
import cv2
import PIL


def canny_fcn(patch: np.array) -> Tuple[np.array, bool]:
    h, w = patch.shape[:2]
    patch_img = PIL.Image.fromarray(patch)
    patch_gray = np.array(patch_img.convert('L'))
    # tile_to_grayscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as grayscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a grayscale, not color.

    # hardcoded thresholds
    edge = cv2.Canny(patch_gray, 40, 100)
    edge = (edge / np.max(edge)) if np.max(edge) != 0 else 0    # avoid dividing by zero
    edge = (np.sum(np.sum(edge)) / (h * w) * 100) if (h * w) != 0 else 0

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    if edge < 2:
        # return a black image + rejected=True
        return np.zeros_like(patch), True
    else:
        # return the patch + rejected=False
        return patch, False


def reject_background(img: np.array, patch_shape: Tuple[int,int], step: int, cores: int = 8) -> \
Tuple[np.ndarray, np.ndarray, List[Any]]:
    patch_shape = np.array(patch_shape)
    h_patches, w_patches = np.ceil(np.array(img.shape)[:2] / patch_shape).astype(int)
    n = h_patches * w_patches
    print(f"\nCanny background rejection...")

    patches_shapes_list = []
    begin = time.time()
    with futures.ThreadPoolExecutor(cores) as executor:
        future_coords: Dict[futures.Future, int] = {}
        for i in range(h_patches):
            for j in range(w_patches):
                k = i*w_patches + j
                patch = img[i*patch_shape[0]:i*patch_shape[0]+step, j*patch_shape[1]:j*patch_shape[1]+step]
                if ((real_shape := np.array(patch.shape[:2])) < patch_shape).any():
                    padding = patch_shape - real_shape
                    patch = np.pad(patch, pad_width=((0, padding[0]), (0, padding[1]), (0, 0)))

                patches_shapes_list.append(patch.shape)
                future = executor.submit(canny_fcn, patch)
                future_coords[future] = k # index 0 - 3. (0,0) = 0, (0,1) = 1, (1,0) = 2, (1,1) = 3
        del img
        
        #num of patches x 224 x 224 x 3 for RGB patches
        tissue_patches = np.zeros((n, patch_shape[0], patch_shape[1], 3), dtype=np.uint8)
        has_tissue = np.zeros(n, dtype=bool)
        for tile_future in futures.as_completed(future_coords):
            k = future_coords[tile_future]
            patch, is_rejected = tile_future.result()
            tissue_patches[k] = patch
            has_tissue[k] = is_rejected

    print(f"Finished Canny background rejection, rejected {np.sum(has_tissue)}/{n} tiles ({time.time()-begin:.2f} seconds)")
    return tissue_patches, has_tissue, patches_shapes_list

from __future__ import division
from concurrent import futures
from typing import Dict, Tuple
import numpy as np
from numba import njit
from tqdm import tqdm


def standardize_brightness(I: np.ndarray) -> np.ndarray:
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I: np.ndarray) -> np.ndarray:
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    """
    I[(I == 0)] = 1
    return I


@njit
def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def RGB_to_OD_nojit(I: np.ndarray) -> np.ndarray:
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    return np.maximum(-1 * np.log(I / 255), 1e-6)


@njit
def OD_to_RGB(OD: np.ndarray) -> np.ndarray:
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    OD = np.maximum(OD, 1e-6)
    return np.clip(255 * np.exp(-OD), 0, 255).astype(np.uint8)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


@njit
def principle_colors(V: np.ndarray, minPhi: float, maxPhi: float):
    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    return v1, v2


@njit
def get_phi(OD: np.ndarray, V: np.ndarray, angular_percentile: int):
    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, 100 - angular_percentile)
    maxPhi = np.percentile(phi, angular_percentile)
    return minPhi, maxPhi


@njit
def get_principle_colors(OD: np.ndarray, V: np.ndarray, angular_percentile: int):
    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, 100 - angular_percentile)
    maxPhi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    return v1, v2


@njit
def calc_hematoxylin(source_concentrations, h, w):
    H = source_concentrations[:, 0].reshape(h, w)
    H = np.exp(-H)
    return H


@njit
def norm_patch(
    source_concentrations: np.ndarray,
    stain_matrix_target: np.ndarray,
    maxC_target: np.ndarray,
    maxC_source: np.ndarray,
    patch_shape: Tuple[int, int, int],
) -> np.ndarray:
    source_concentrations *= maxC_target / maxC_source
    return np.clip(
        255 * np.exp(-np.dot(source_concentrations, stain_matrix_target).reshape(patch_shape)),
        0, 255
    ).astype(np.uint8)


def norm_patch_fn(
    src_concentrations: np.ndarray,
    stain_matrix_target: np.ndarray,
    maxC_target: np.ndarray,
    patch_shape: np.ndarray,
) -> np.ndarray:
    maxC_source = np.percentile(src_concentrations, 99, axis=0)[None]
    patch_normed = norm_patch(src_concentrations, stain_matrix_target, maxC_target, maxC_source, patch_shape)
    return patch_normed


def get_target_concentrations(arr: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    arr = remove_zeros(arr)
    OD = RGB_to_OD_nojit(arr).reshape((-1, 3)) # runs ~2x faster without jit :/
    try:
        # stain_matrix.T @ x = OD.T
        x, *_ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
        x = x.T
    except Exception as e:
        print(e)
        x = None
    return x


def get_src_concentration(
        patches_flat: np.ndarray, stain_matrix: np.ndarray, cores: int = 8
) -> np.ndarray:
    print(f"Normalizing {patches_flat.shape[0]} tiles...")
    n, pxls = patches_flat.shape[0], patches_flat.shape[1] * patches_flat.shape[2]
    src_concentrations = np.zeros((n, pxls, 2), dtype=np.float64)

    with futures.ThreadPoolExecutor(cores) as executor:
        future_coords: Dict[futures.Future, int] = {}
        for k, patch in enumerate(patches_flat):
            future = executor.submit(get_target_concentrations, patch, stain_matrix)
            future_coords[future] = k

        for tile_future in tqdm(
            futures.as_completed(future_coords),
            total=patches_flat.shape[0],
            desc="Calculating concentrations",
            leave=False,
        ):
            k = future_coords[tile_future]
            src_concentrations[k] = tile_future.result()

    return src_concentrations

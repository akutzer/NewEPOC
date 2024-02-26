"""
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

from concurrent import futures
from typing import Dict
import time

import numpy as np
from numba import njit
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm


##########################################

def read_image(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def show_colors(C):
    """
    Shows rows of C as colors (RGB)
    :param C:
    :return:
    """
    n = C.shape[0]
    for i in range(n):
        if C[i].max() > 1.0:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255, linewidth=20)
        else:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])


def show(image, now=True, fig_size=(10, 10)):
    """
    Show an image (np.array).
    Caution! Rescales image to be in range [0,1].
    :param image:
    :param now:
    :param fig_size:
    :return:
    """
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()


def build_stack(tup):
    """
    Build a stack of images from a tuple of images
    :param tup:
    :return:
    """
    N = len(tup)
    if len(tup[0].shape) == 3:
        h, w, c = tup[0].shape
        stack = np.zeros((N, h, w, c))
    if len(tup[0].shape) == 2:
        h, w = tup[0].shape
        stack = np.zeros((N, h, w))
    for i in range(N):
        stack[i] = tup[i]
    return stack


def patch_grid(ims, width=5, sub_sample=None, rand=False, save_name=None):
    """
    Display a grid of patches
    :param ims:
    :param width:
    :param sub_sample:
    :param rand:
    :return:
    """
    N0 = np.shape(ims)[0]
    if sub_sample == None:
        N = N0
        stack = ims
    elif sub_sample != None and rand == False:
        N = sub_sample
        stack = ims[:N]
    elif sub_sample != None and rand == True:
        N = sub_sample
        idx = np.random.choice(range(N), sub_sample, replace=False)
        stack = ims[idx]
    height = np.ceil(float(N) / width).astype(np.uint16)
    plt.rcParams['figure.figsize'] = (18, (18 / width) * height)
    plt.figure()
    for i in range(N):
        plt.subplot(height, width, i + 1)
        im = stack[i]
        show(im, now=False, fig_size=None)
    if save_name != None:
        plt.savefig(save_name)
    plt.show()


######################################

def standardize_brightness(I):
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    """
    mask = (I == 0)
    I[mask] = 1
    return I

@njit
def RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    return np.maximum(-1 * np.log(I / 255), 1e-6)

@njit
def OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


@njit
def norm_patch(source_concentrations, stain_matrix_target, maxC_target, maxC_source, patch_shape):
    source_concentrations *= (maxC_target / maxC_source)
    return (255 * np.exp(-np.dot(source_concentrations, stain_matrix_target).reshape(patch_shape))).astype(np.uint8)


def norm_patch_fn(src_concentrations, stain_matrix_target, maxC_target, patch_shape):
    maxC_source = np.percentile(src_concentrations, 99, axis=0)[None]
    jit_output = norm_patch(src_concentrations, stain_matrix_target, maxC_target, maxC_source, patch_shape)
    return(jit_output)


def get_target_concentrations(arr, stain_matrix):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    arr = remove_zeros(arr)
    OD = RGB_to_OD(arr).reshape((-1, 3))
    try:
        #limited Lasso to 1 thread, instead of taking all available threads (-1 default)
        temp, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
        temp=temp.T
    except Exception as e:
        print(e)
        temp = None
    return temp


def get_src_concentration(patches_flat, stain_matrix, rejection_arr, cores: int=8):
    print(f"Normalising {np.sum(~rejection_arr)} tiles...")
    n, pxls = patches_flat.shape[0], patches_flat.shape[1] * patches_flat.shape[2]
    src_concentrations = np.zeros((n, pxls, 2), dtype=np.float64)

    with futures.ThreadPoolExecutor(cores) as executor:
        future_coords: Dict[futures.Future, int] = {}
        for k, patch in enumerate(patches_flat):
            if not rejection_arr[k]:
                future = executor.submit(get_target_concentrations, patch, stain_matrix)
                future_coords[future] = k        

        for tile_future in tqdm(futures.as_completed(future_coords), total=np.sum(~rejection_arr), desc='Normalizing tiles', leave=False):
            k = future_coords[tile_future]
            src_concentrations[k] = tile_future.result()
  
    return src_concentrations


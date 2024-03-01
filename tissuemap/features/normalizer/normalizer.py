"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

For the original implementation see: https://github.com/Peter554/StainTools
"""
from __future__ import division
import time
from concurrent import futures
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from . import utils


class MacenkoNormalizer:
    """
    A stain normalization using the methods proposed by Macenko et al.
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None

    def fit(self, target):
        target = utils.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix(target)
        self.target_concentrations = utils.get_target_concentrations(
            target, self.stain_matrix_target
        )
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0)[None]

    def transform(
        self, slide_array: np.ndarray, patches: np.ndarray, cores: int = 8, legacy_norm: bool = False
    ) -> np.ndarray:
        """Returns an array the same shape as `patches` with Macenko normalization applied to all patches."""
        start_normalizing = time.time()
        # calculates the stain matrix only from the patches that contain some tissue
        # to restore the old behavior comment out the following line and uncomment the line after that
        if legacy_norm:
            stain_matrix_src = self.get_stain_matrix(slide_array) # shape: (2, 3)
        else:
            stain_matrix_src = self.get_stain_matrix(patches) # shape: (2, 3)
        print(
            f"Get stain matrix ({(after_stain_mat := time.time()) - start_normalizing:.2f} seconds)"
        )

        src_concentrations = utils.get_src_concentration(
            patches, stain_matrix_src, cores
        ) # shape: (n_patches, patch_h*patch_w, 3)
        del stain_matrix_src
        print(
            f" Get concentrations for normalization ({(after_conc := time.time()) - after_stain_mat:.2f} seconds)"
        )

        norm_patches = self._norm_patches(src_concentrations, patches.shape[1:], cores)
        # shape: (n_patches, patch_h, patch_w, 3)
        print(
            f" Normalized {len(patches)} patches ({time.time()-after_conc:.2f} seconds)"
        )
        return norm_patches

    def _norm_patches(
        self,
        src_concentrations: np.ndarray,
        patch_shape: Tuple[int, int, int],
        cores: int = 8,
    ) -> np.ndarray:
        n = src_concentrations.shape[0]
        with futures.ThreadPoolExecutor(cores) as executor:
            future_coords: Dict[futures.Future, int] = {}
            for i, src_conc in enumerate(src_concentrations):
                future = executor.submit(
                    utils.norm_patch_fn,
                    src_conc,
                    self.stain_matrix_target,
                    self.maxC_target,
                    patch_shape=patch_shape,
                )
                future_coords[future] = i

            norm_patches = np.zeros((n, *patch_shape), dtype=np.uint8)
            for tile_future in tqdm(
                futures.as_completed(future_coords),
                total=n,
                desc="Normalizing patches",
                leave=False,
            ):
                i = future_coords[tile_future]
                patch = tile_future.result()
                norm_patches[i] = patch
        return norm_patches

    def target_stains(self):
        return utils.OD_to_RGB(self.stain_matrix_target)

    @staticmethod
    def get_stain_matrix(
        I: np.ndarray, luminosity_threshold: float = 0.15, angular_percentile: int = 99
    ) -> np.ndarray:
        """
        Stain matrix estimation via method of:
        M. Macenko et al. 'A method for normalizing histology slides for quantitative analysis'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return:
        """
        # Convert to OD and ignore background
        I = utils.remove_zeros(I)
        OD = utils.RGB_to_OD_nojit(I).reshape(-1, 3)
        OD = OD[(OD > luminosity_threshold).any(axis=1)] # main bottleneck of this function
        # tried using np.ma.masked_array, but this slowed down the following code

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        V = np.linalg.eigh(np.cov(OD, rowvar=False)).eigenvectors

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1

        # Project on this basis.
        That = np.dot(OD, V)

        # # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])

        # # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return utils.normalize_rows(HE)

    def hematoxylin(self, I: np.ndarray) -> np.ndarray:
        I = utils.standardize_brightness(I)
        h, w = I.shape[:2]
        stain_matrix_source = self.get_stain_matrix(I)
        source_concentrations = utils.get_target_concentrations(
            I, stain_matrix_source
        )  # put target here, just in case

        del I, stain_matrix_source

        H = utils.calc_hematoxylin(source_concentrations, h, w)
        return H

__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"

import os
from pathlib import Path
import logging
import time
from typing import Optional
import random

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from tissuemap.features.extractor.feature_extractors import FeatureExtractor, store_features, store_metadata
from tissuemap.classifier.model import HistoClassifier
from tissuemap.features.helpers.common import supported_extensions

Image.MAX_IMAGE_PIXELS = None



def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, classifier_model_path: Path,
               cache_dir: Path, cache: bool = False, norm: bool = False, normalization_template: Optional[Path] = None,
               del_slide: bool = False, only_feature_extraction: bool = False,
               keep_dir_structure: bool = False, cores: int = 8, target_microns: int = 256,
               patch_size: int = 224, batch_size: int = 64, device: str = "cuda"
               ):
   
    # Initialize the feature extraction model
    print(f"Initializing CTransPath model as feature extractor...")
    has_gpu = torch.cuda.is_available()
    device = torch.device(device) if "cuda" in device and has_gpu else torch.device("cpu")
    extractor = FeatureExtractor.from_checkpoint(checkpoint_path=model_path, device=device)
    patch_classifier = HistoClassifier.from_pretrained(classifier_model_path, device=device)

    # Create output and cache directories
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_"
    model_name_norm = Path(norm_method + extractor.model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    # Get list of slides, filter out slides that have already been processed
    print("Scanning for existing feature files...")
    existing = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []
    img_dir = [svs for ext in supported_extensions for svs in wsi_dir.glob(f"**/*{ext}")]
    existing = [f for f in existing if f in [f.stem for f in img_dir]]
    img_dir = [f for f in img_dir if f.stem not in existing]

    random.shuffle(img_dir)
    num_processed = 0
    
    batch_size = 10_000
    batch_patches = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
    batch_classes = []

    embeddings, classes = [], [] 

    for k, slide_url in enumerate(tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0, position=1)):
        # if "DEB" in slide_url.parent.stem:
        #     print(slide_url.parent.stem)
        #     continue
        slide_name = slide_url.stem

        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not (os.path.exists((f"{feat_out_dir}.h5"))) and not os.path.exists(f"{slide_url}.tmp"):
            patches = np.array(Image.open(slide_url))
            batch_patches[k % batch_size] = patches
            batch_classes.append(slide_url.parent.stem)

            if (k+1) % batch_size == 0:
                store_metadata(
                    outdir=feat_out_dir,
                    extractor_name=extractor.name,
                    patch_size=patch_size,
                    target_microns=target_microns,
                    normalized=norm
                )
                features = extractor.extract(batch_patches, cores, 64)
                patch_cls = patch_classifier.predict_patches(batch_patches, cores, 64)

                batch_classes = np.array(batch_classes)[..., None]
                patch_cls["probability"] = 0.
                patch_cls["probability"][patch_cls["label"] == batch_classes] = 1.
                patch_cls.sort(order='probability', axis=-1)
                patch_cls = patch_cls[..., ::-1]

                embeddings.append(features)
                classes.append(patch_cls)

                # feat_out_dir = feat_out_dir.parent / f"NCT_embeddings_{(k+1) // batch_size}"
                # store_features(feat_out_dir, features, patch_cls, np.zeros((1, 4)), extractor.name, patch_classifier.config.categories)

                batch_classes = []
                batch_patches.fill(0)
                num_processed += 1


        else:
            if os.path.exists((f"{feat_out_dir}.h5")):
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            existing.append(slide_name)
    

    embeddings, classes = np.concatenate(embeddings), np.concatenate(classes)
    print(embeddings.shape, classes.shape)
    feat_out_dir = feat_out_dir.parent / f"NCT_embeddings"
    store_features(feat_out_dir, embeddings, classes, np.zeros((1, 4)), extractor.name, patch_classifier.config.categories)

    
    

if __name__ == "__main__":

    output_dir= Path("/home/aaron/work/EKFZ/data/NCT-CRC-HE/features-RENORM")
    wsi_dir= Path("/home/aaron/work/EKFZ/data/NCT-CRC-HE/NCT-CRC-HE-100K-RENORM")
    model_path = Path("/home/aaron/work/EKFZ/tissueMAP/tissuemap/ressources/ctranspath.pth")
    classifier_model_path = Path("/home/aaron/work/EKFZ/tissueMAP/models/swinv2-tiny-patch4-window8-256_binary=False_2024-03-01T15:03:37.995")
    cache_dir = Path("/home/aaron/work/EKFZ/data/NCT-CRC-HE/cache-norm")
    normalization_template = Path("/home/aaron/work/EKFZ/tissueMAP/tissuemap/ressources/normalization_template.jpg")

    preprocess(
        output_dir,
        wsi_dir,
        model_path,
        classifier_model_path,
        cache_dir=cache_dir,
        norm=False,
        del_slide=False,
        only_feature_extraction=False,
        normalization_template=normalization_template,
        device="cuda",
        cache=False,
        cores=16
        # batch_size=16
    )

    

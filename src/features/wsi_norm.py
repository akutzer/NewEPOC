__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"


from pathlib import Path
from contextlib import contextmanager
import logging
import os
import openslide
from tqdm import tqdm
from PIL import Image
import cv2
import time
from datetime import timedelta
from pathlib import Path
from random import shuffle
import torch
from .helpers.stainNorm_Macenko import MacenkoNormalizer
from .helpers.common import supported_extensions
from .helpers.concurrent_canny_rejection import reject_background
from .helpers.loading_slides import load_slide, load_slide_jpg, get_raw_tile_list
from .helpers.feature_extractors import FeatureExtractor, extract_features_
from .helpers.exceptions import MPPExtractionError
from .helpers.utils import reconstruct_from_patches
import numpy as np

Image.MAX_IMAGE_PIXELS = None


@contextmanager
def lock_file(slide_path: Path):
    try:
        Path(f"{slide_path}.tmp").touch()
    except PermissionError:
        pass # No write permissions for wsi directory
    try:
        yield
    finally:
        if os.path.exists(f"{slide_path}.tmp"): # Catch collision cases
            os.remove(f"{slide_path}.tmp")


def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{time.time()}.tmp"
        Path(testfile).touch()
    except PermissionError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)


def save_image(image, path: Path):
    width, height = image.size
    if width > 65500 or height > 65500:
        logging.warning(f"Image size ({width}x{height}) exceeds maximum size of 65500x65500, resizing {path.name} before saving...")
        ratio = 65500 / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)))
    image.save(path)


def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, cache_dir: Path, norm: bool,
               del_slide: bool, only_feature_extraction: bool, cache: bool = True, cores: int = 8,
               target_microns: int = 256, patch_size: int = 224, keep_dir_structure: bool = False,
               device: str = "cuda", normalization_template: Path = None):
    has_gpu = torch.cuda.is_available()
    target_mpp = target_microns/patch_size
    patch_shape = (patch_size, patch_size) #(224, 224) by default
    step_size = patch_size #have 0 overlap by default

    # Initialize the feature extraction model
    print(f"Initialising CTransPath model as feature extractor...")
    extractor = FeatureExtractor()
    model, model_name = extractor.init_feat_extractor(checkpoint_path=model_path, device=device)

    # Create cache and output directories
    if cache:
        cache_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)

    # Create logfile and set up logging
    logfile_name = "logfile_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = output_file_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    logging.info(f"Model: {model_name}\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")
    test_wsidir_write_permissions(wsi_dir)

    if norm:
        print("\nInitialising Macenko normaliser...")
        normalizer = MacenkoNormalizer()
        print(f"Template: {normalization_template}")
        target = Image.open(normalization_template).convert('RGB')
        normalizer.fit(np.array(target))  

    total_start_time = time.time()
    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"

    # Get list of slides, filter out slides that have already been processed
    print("Scanning for existing feature files...")
    existing = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []
    if not only_feature_extraction:
        img_dir = [svs for ext in supported_extensions for svs in wsi_dir.glob(f"**/*{ext}")]
        existing = [f for f in existing if f in [f.stem for f in img_dir]]
        img_dir = [f for f in img_dir if f.stem not in existing]
    else:
        if not cache_dir.exists():
            logging.error("Cache directory does not exist, cannot extract features from cached slides!")
            exit(1)
        img_dir = [jpg for jpg in cache_dir.glob(f"**/*/{img_name}")]
        existing = [f for f in existing if f in [f.parent.name for f in img_dir]]
        img_dir = [f for f in img_dir if f.parent.name not in existing]

    shuffle(img_dir)
    num_total = len(img_dir) + len(existing)
    num_processed = 0
    error_slides = []
    if len(existing):
        print(f"\n For {len(existing)} out of {num_total} slides in the wsi directory feature files were found, skipping these slides...")
    for slide_url in tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem if not only_feature_extraction else slide_url.parent.name
        slide_cache_dir = cache_dir/slide_name
        if cache:
            slide_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n\n===== Processing slide {slide_name} =====")
        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not (os.path.exists((f"{feat_out_dir}.h5"))) and not os.path.exists(f"{slide_url}.tmp"):
            with lock_file(slide_url):
                if (
                    (only_feature_extraction and (slide_jpg := slide_url).exists()) or \
                    (slide_jpg := slide_cache_dir/"norm_slide.jpg").exists()
                ):
                    patches, patches_coords, n = load_slide_jpg(slide_jpg, patch_shape)
                    print(f"Loaded {img_name}, {len(patches)}/{n} tiles remain")
                else:
                    try:
                        slide = openslide.OpenSlide(slide_url)
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error("Unsupported format for slide, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue

                    start_loading = time.time()
                    try:
                        slide_array = load_slide(slide=slide, target_mpp=target_mpp, cores=cores)
                    except MPPExtractionError:
                        if del_slide:
                            logging.error("MPP missing in slide metadata, deleting slide and continuing...")
                            if os.path.exists(slide_url):
                                os.remove(slide_url)
                        else:
                            logging.error("MPP missing in slide metadata, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except openslide.lowlevel.OpenSlideError as e:
                        print("")
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue
                    del slide   # Remove .SVS from memory

                    print(f" Loaded slide ({time.time() - start_loading:.2f} seconds)")
                    print(f"Size of WSI: {slide_array.shape}")
                        
                    if cache:   # Save raw .svs jpg
                        raw_image = Image.fromarray(slide_array)
                        save_image(raw_image, slide_cache_dir/"slide.jpg")

                    # Do edge detection here and reject unnecessary tiles BEFORE normalization
                    patches, patch_rejected, patch_shapes = reject_background(
                        img=slide_array, patch_shape=patch_shape, step=step_size, cores=cores
                    )
                    if cache:
                        print("Reconstructing canny img")
                        canny_img = reconstruct_from_patches(patches, slide_array.shape[:2])
                        save_image(canny_img, slide_cache_dir/"canny_slide.jpg")

                    # Pass raw slide_array for getting the initial concentrations, tissue_patches for actual normalization
                    if norm:
                        print(f"\nNormalizing slide...")
                        start_normalizing = time.time()                        
                        stain_matrix = normalizer.get_stain_matrix(slide_array)
                        print(f"Get stain matrix ({time.time()-start_normalizing:.2f} seconds)")
                        norm_patches = normalizer.transform(patches, stain_matrix, patch_rejected)                        
                        print(f"Normalized slide ({time.time() - start_normalizing:.2f} seconds)")
                        if cache:
                            norm_img = reconstruct_from_patches(norm_patches, slide_array.shape[:2])
                            save_image(norm_img, slide_cache_dir/"norm_slide.jpg")
                    else:
                        canny_img, patches, patches_coords = get_raw_tile_list(slide_array.shape, patches,
                                                                                        patch_rejected, patch_shapes)

                    if cache:
                        print("Saving Canny background rejected image...")
                        save_image(canny_img, slide_cache_dir/"canny_slide.jpg")

                    # Remove original slide jpg from memory
                    del slide_array
                    
                    # Optionally remove the original slide from harddrive
                    if del_slide:
                        print("Deleting slide from local folder...")
                        if os.path.exists(slide_url):
                            os.remove(slide_url)

                print("\nExtracting CTransPath features from slide...")
                start_time = time.time()
                if len(patches) > 0 and False:
                    extract_features_(model=model, model_name=model_name, norm_wsi_img=patches,
                                    coords=patches_coords, wsi_name=slide_name, outdir=feat_out_dir, cores=cores,
                                    is_norm=norm, device=device if has_gpu else "cpu", target_microns=target_microns,
                                    patch_size=patch_size)
                    logging.info(f" Extracted features from slide: {time.time() - start_time:.2f} seconds ({len(patches)} tiles)")
                    num_processed += 1
                else:
                    logging.error(" 0 tiles remain to extract features from after pre-processing. Continuing...")
                    error_slides.append(slide_name)
                    continue
        else:
            if os.path.exists((f"{feat_out_dir}.h5")):
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            existing.append(slide_name)
            if del_slide:
                print("Deleting slide from local folder...")
                if os.path.exists(slide_url):
                    os.remove(slide_url)

    logging.info(f"===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    logging.info(f"Summary: Processed {num_processed} slides, encountered {len(error_slides)} errors, skipped {len(existing)} readily-processed slides")
    if len(error_slides):
        logging.info("The following slides were not processed due to errors:\n\n" + "\n".join(error_slides))

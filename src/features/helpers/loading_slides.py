import re
from typing import Dict, Tuple
from concurrent import futures
import openslide
from tqdm import tqdm
import numpy as np
import PIL

from .exceptions import MPPExtractionError

PIL.Image.MAX_IMAGE_PIXELS = None



def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224, cores: int = 8, chunks: int = 8) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process using Threads since it's IO heavy
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    stride = np.ceil(np.array(slide.dimensions) / chunks).astype(int)
    slide_mpp = float(get_slide_mpp(slide))
    # (width, height) for openslide
    tile_size = np.round(stride * slide_mpp / target_mpp).astype(int)

    with futures.ThreadPoolExecutor(cores) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(chunks):  # row
            for j in range(chunks):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an array as soon as they are loaded
        n_tiles_w, n_tiles_h = tile_size * chunks
        img = np.zeros((n_tiles_h, n_tiles_w, 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=chunks*chunks, desc='Reading WSI tiles', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_size * (j, i)    # switch (w,h) to (h,w) for numpy
            img[y:y+tile_size[1], x:x+tile_size[0], :] = tile
    return img


def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {slide_mpp}")
    except KeyError:
        # Try out the missing MPP handlers
        try:
            slide_mpp = extract_mpp_from_comments(slide)
            if slide_mpp:
                print(f"MPP retrieved from comments after initial failure: {slide_mpp}")
            else:
                print(f"MPP is missing in the comments of this file format, attempting to extract from metadata...")
                slide_mpp = extract_mpp_from_metadata(slide)
                print(f"MPP re-matched from metadata after initial failure: {slide_mpp}")
        except:
            raise MPPExtractionError("MPP could not be loaded from the slide!")
    return slide_mpp


def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    import xml.dom.minidom as minidom
    xml_path = slide.properties['tiff.ImageDescription']
    doc = minidom.parseString(xml_path)
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    return mpp


def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    slide_properties = slide.properties.get('openslide.comment')
    pattern = r'<PixelSizeMicrons>(.*?)</PixelSizeMicrons>'
    match = re.search(pattern, slide_properties)
    if match:
        return float(match.group(1))
    else:
        return None


def load_slide_jpg(slide_jpg: PIL.Image, patch_shape: Tuple[int, int]):
    img = PIL.Image.open(slide_jpg)
    arr = np.array(img)
    
    h_stride, w_stride = patch_shape
    h_patches, w_patches = np.array(arr.shape[:2]) // patch_shape
    n = h_patches * w_patches

    patches, patch_coords = [], []
    for i in range(h_patches):
        for j in range(w_patches):
            x, y = i*h_stride, j*w_stride   # (height-coord, width-coord)
            patch = arr[x:x+h_stride, y:y+w_stride, :]
            # if patch is not fully black (i.e. rejected previously)
            if np.any(patch):
                patches.append(patch)
                patch_coords.append((x, y))
    return patches, patch_coords, n


def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = PIL.Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(PIL.Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))

    return canny_img, canny_output_array, coords_list


# test get_raw_tile_list function
def test_get_raw_tile_list():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    canny_img, canny_patch_list, coords_list = get_raw_tile_list(img.shape, img, img, (224,224))
    assert len(canny_patch_list) == 4
    assert len(coords_list) == 4
    assert canny_patch_list[0].shape == (224,224,3)
    assert coords_list[0] == (0,0)

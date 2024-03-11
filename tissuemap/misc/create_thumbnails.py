import os
from pathlib import Path
import logging
from tqdm import tqdm
import openslide


wsi_dir = Path("/mnt/bulk/aaron/NewEPOC/data/NewEPOC")
cache_dir = wsi_dir/"thumbnails"
cache_dir.mkdir(parents=True, exist_ok=True)

existing = [f.stem for f in cache_dir.glob("**/*.jpg")]
print(f"Existing: {len(existing)}")
svs_files = [svs for svs in wsi_dir.glob(f"**/*.svs") if svs.stem not in existing]
print(f"To process: {len(svs_files)}")

for slide_path in tqdm(svs_files):
    print(slide_path)
    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        logging.error(f"Unsupported format for slide {slide_path}, continuing...")
        continue
    except Exception as e:
        logging.error(f"Failed loading slide {slide_path}, continuing... Error: {e}")
        continue
    thumbnail = slide.get_thumbnail((3072, 3072))
    thumbnail.save(cache_dir/f"{slide_path.stem}.jpg")
    
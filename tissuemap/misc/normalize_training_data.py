from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

from tissuemap.features.normalizer.normalizer import MacenkoNormalizer



def renorm_dataset(dataset_path: Path, normalization_template: Path, batch_size: int = 5_000):
    norm_dataset_path = dataset_path.parent / (dataset_path.name + "-RENORM")
    norm_dataset_path.mkdir(parents=True, exist_ok=True)

    normalizer = MacenkoNormalizer()
    target = Image.open(normalization_template).convert('RGB')
    normalizer.fit(np.array(target))  
    tile_paths = list(dataset_path.glob("**/*.tif"))
    random.shuffle(tile_paths)


    for i in tqdm(range(0, len(tile_paths), batch_size)):
        tile_batch = tile_paths[i: i+batch_size]
        tiles = np.zeros((len(tile_batch), 224, 224, 3))
        for i, tile_path in enumerate(tqdm(tile_batch, leave=False)):
            tiles[i] = np.array(Image.open(tile_path).convert('RGB'))
        norm_tiles = normalizer.transform(tiles)

        for tile_path, norm_tile in zip(tile_batch, norm_tiles):
            norm_tile_path = norm_dataset_path / tile_path.parent.relative_to(dataset_path)
            norm_tile_path.mkdir(parents=True, exist_ok=True)
            norm_tile_path = norm_tile_path / (tile_path.stem + "_renorm" + tile_path.suffix)
            norm_tile = Image.fromarray(norm_tile)
            norm_tile.save(norm_tile_path) 



if __name__ == "__main__":
    normalization_template = Path("/home/aaron/work/EKFZ/tissueMAP/tissuemap/ressources/normalization_template.jpg")
    dataset_path = Path("/home/aaron/work/EKFZ/data/NCT-CRC-HE/NCT-CRC-HE-100K")
    # dataset_path = Path("/home/aaron/work/EKFZ/data/NCT-CRC-HE/CRC-VAL-HE-MERGED")
    
    
    renorm_dataset(dataset_path, normalization_template, batch_size=5_000)
    




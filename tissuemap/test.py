from pathlib import Path
import tracemalloc
from features.wsi_norm import preprocess
# from marugoto_norm.wsi_norm import preprocess




output_dir= Path("/home/aaron/work/EKFZ/data/NewEPOC/features")
wsi_dir= Path("/home/aaron/work/EKFZ/data/NewEPOC/wsi")
model_path = Path("/home/aaron/work/EKFZ/tissueMAP/tissuemap/ressources/ctranspath.pth")
classifier_model_path = Path("/home/aaron/work/EKFZ/tissueMAP/models/swinv2-tiny-patch4-window8-256_binary=False_2024-03-07T15:53:19.091")
cache_dir = Path("/home/aaron/work/EKFZ/data/NewEPOC/cache")
normalization_template = Path("/home/aaron/work/EKFZ/tissueMAP/tissuemap/ressources/normalization_template.jpg")


tracemalloc.start()

preprocess(
    output_dir,
    wsi_dir,
    model_path,
    classifier_model_path,
    cache_dir=cache_dir,
    cache=True,
    norm=True,
    del_slide=False,
    only_feature_extraction=False,
    normalization_template=normalization_template,
    device="cuda",
    # batch_size=16
)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()
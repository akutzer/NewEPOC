from pathlib import Path
import tracemalloc
from features.wsi_norm import preprocess
# from marugoto_norm.wsi_norm import preprocess




output_dir= Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/")
wsi_dir= Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC/")
model_path = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/src/ressources/ctranspath.pth")
classifier_model_path = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/models/efficientnet-b0_binary=False_2024-02-24T11:48:56.609")
cache_dir = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/cache/")
normalization_template = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/src/ressources/normalization_template.jpg")


tracemalloc.start()

preprocess(
    output_dir,
    wsi_dir,
    model_path,
    classifier_model_path,
    cache_dir=cache_dir,
    norm=True,
    del_slide=False,
    only_feature_extraction=False,
    normalization_template=normalization_template,
    device="cuda",
    # batch_size=16
)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()
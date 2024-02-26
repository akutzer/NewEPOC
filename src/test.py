from pathlib import Path
from features.wsi_norm import preprocess



output_dir= Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/")
wsi_dir= Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC/")
model_path = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/src/ressources/ctranspath.pth")
cache_dir = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/cache/")
normalization_template = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/src/ressources/normalization_template.jpg")

preprocess(
    output_dir,
    wsi_dir,
    model_path,
    cache_dir,
    norm=True,
    del_slide=False,
    only_feature_extraction=False,
    normalization_template=normalization_template,
    device="cpu"
)
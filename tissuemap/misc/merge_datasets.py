from pathlib import Path
from distutils.dir_util import copy_tree
import shutil
import random
from tqdm import tqdm


data_dir = Path("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/tissueMAP/data/")

train_old_norm_dir = data_dir / "NCT-CRC-HE-100K"
train_renorm_dir = data_dir / "NCT-CRC-HE-100K-RENORM"
train_nonorm_dir = data_dir / "NCT-CRC-HE-100K-NONORM"


val_old_norm_dir = data_dir / "CRC-VAL-HE-7K"
val_renorm_dir = data_dir / "CRC-VAL-HE-7K-RENORM"

train_merged_dir = data_dir / "NCT-CRC-HE-MERGED"
train_merged_dir.mkdir(parents=True, exist_ok=True)
val_merged_dir = data_dir / "CRC-VAL-HE-MERGED"
val_merged_dir.mkdir(parents=True, exist_ok=True)



# copy train 
copy_tree(str(train_old_norm_dir), str(train_merged_dir))
train_renorm_files = list(train_renorm_dir.glob("**/*.tif"))
train_renorm_renamed = list(map(lambda path: str(train_merged_dir / path.relative_to(train_renorm_dir)).replace(".tif", "_renormed.tif"), train_renorm_files))
for src, dst in tqdm(zip(train_renorm_files, train_renorm_renamed), total=len(train_renorm_renamed)):
    shutil.copy2(src, dst)


# copy val 
copy_tree(str(val_old_norm_dir), str(val_merged_dir))
val_renorm_files = list(val_renorm_dir.glob("**/*.tif"))
val_renorm_renamed = list(map(lambda path: str(val_merged_dir / path.relative_to(val_renorm_dir)).replace(".tif", "_renormed.tif"), val_renorm_files))
for src, dst in tqdm(zip(val_renorm_files, val_renorm_renamed), total=len(val_renorm_renamed)):
    shutil.copy2(src, dst)


# split nonorm
nonorm_files = list(train_nonorm_dir.glob("**/*.tif"))
nonorm_val = random.sample(nonorm_files, k=len(nonorm_files) // 8)
nonorm_train = list(set(nonorm_files) - set(nonorm_val))
print(len(nonorm_files), len(nonorm_train), len(nonorm_val))
nonorm_val_newnames = list(map(lambda path: val_merged_dir / path.relative_to(train_nonorm_dir), nonorm_val))
nonorm_tain_newnames = list(map(lambda path: train_merged_dir / path.relative_to(train_nonorm_dir), nonorm_train))

for src, dst in tqdm(zip(nonorm_train, nonorm_tain_newnames), total=len(nonorm_tain_newnames)):
    shutil.copy2(src, dst)
for src, dst in tqdm(zip(nonorm_val, nonorm_val_newnames), total=len(nonorm_val_newnames)):
    shutil.copy2(src, dst)


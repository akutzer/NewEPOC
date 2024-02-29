from PIL import Image
import numpy as np
import h5py
from pathlib import Path

# Image.MAX_IMAGE_PIXELS = None

# img_old = Image.open("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/cache/418752/norm_slide (1).jpg")
# img_new = Image.open("/home/aaron/Documents/Studium/Informatik/7_Semester/EKFZ/NewEPOC/data/NewEPOC-features/cache/418752/norm_slide.jpg")
# img_old = np.array(img_old)
# img_new = np.array(img_new)



# # print(np.linalg.norm(img_old - img_new, axis=(0,1)))
# # print(np.sum(diff).shape, img_new.size)
# # print(np.sum(diff) / img_new.size)

# print(np.sum(img_old != img_new) / img_new.size)

# Image.fromarray((img_old != img_new).any(axis=-1)).save("diff.jpg")
# Image.fromarray(img_old - img_new).save("diff_2.jpg")


# new = Path("data/NewEPOC-features/STAMP_macenko_xiyuewang-ctranspath-7c998680/418752.h5")
# old = Path("data/NewEPOC-features/STAMP_macenko_xiyuewang-ctranspath-7c998680/418752 (1).h5")

new = Path("data/NewEPOC-features/STAMP_macenko_xiyuewang-ctranspath-7c998680/418752.h5")
old = Path("data/NewEPOC-features/STAMP_macenko_xiyuewang-ctranspath-7c998680/418752 (1).h5")


with h5py.File(old, 'r') as f_old, h5py.File(new, 'r') as f_new:
    print(f_old.keys())
    print(f_new.keys())
    x, y = f_old["feats"][:], f_new["feats"][:]
    diff = x-y
    diff_axis = np.max(diff, axis=-1)
    print(np.sort(diff_axis))
    print(np.allclose(x, y, atol=1e-3))
    print((x == y).all())



# [0.000e+00 0.000e+00 9.537e-07 ... 1.953e-03 2.441e-03 3.174e-03]
# False
# False


# [0.0476  0.05286 0.05566 ... 0.9897  1.236   1.303  ]
# False
# False

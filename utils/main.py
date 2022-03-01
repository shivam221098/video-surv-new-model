import cv2
import glob
import shutil
import os
import numpy as np

# filenames = glob.glob("*/Train/*/*.*")
# try:
#     os.mkdir("NonViolence")
# except FileExistsError:
#     pass
#
# index = 1022
# for filename in filenames:
#     shutil.move(filename, f"NonViolence/NV_{index}.tif")
#     index += 1

x = cv2.imread("D:/Python Programs/vid-survi-20-feb/dataset/Real Life Violence Dataset/NonViolence/NV_7108.tif")
print(cv2.resize(x, (64, 64)).shape)
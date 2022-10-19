import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from augment import *


augment_folder = "./result/"
original_folder = "./images-20221018T171551Z-001/images/"
i = 0


def crop_image(img_array):
    h_min, h_max, w_min, w_max, _ = text_localization(img_array)
    return img_array[h_min:h_max, w_min:w_max]


for image in os.listdir(original_folder)[:1000]:
    img = cv2.imread(os.path.join(original_folder, image))
    crop_img = crop_image(img)
    # cv2.imshow("HIEU", crop_img)
    # cv2.waitKey(0)
    # plt.imshow(crop_img)
    # plt.show()
    cv2.imwrite(os.path.join(augment_folder, f"{i}.png"), crop_img)
    i += 1

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(
    #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)
    # freq, value = np.histogram(thresh, bins=np.arange(256))
    # plt.plot(freq)
    # plt.show()
    # pd.DataFrame(freq).to_csv("freq.csv")

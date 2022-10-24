import cv2
import numpy as np
import os


def text_localization(img_array):
    width, height = img_array.shape[1], img_array.shape[0]
    brown_lo = np.array([0])
    # brown_hi = np.array([img_array.max() - 90])
    brown_hi = np.array([150])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
    hsv = cv2.GaussianBlur(hsv, (9, 9), 0)
    hsv = cv2.adaptiveThreshold(
        hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    mask = cv2.inRange(hsv, brown_lo, brown_hi)
    h, w = np.where(mask > 0)

    cond = (h < height - 20) & (w < width - 20) & (h > 20) & (w > 20)
    h_min = np.min(h[cond])
    h_max = np.max(h[cond])

    w_min = np.min(w[cond])
    w_max = np.max(w[cond])

    color = img_array[h[0], w[0]]
    color = (int(color[0]), int(color[1]), int(color[2]))
    # cv2.imshow("HIEU", cv2.rectangle(img_array, (w_min, h_min), (w_max, h_max),
    #  color = (0, 0, 255), thickness = 2))
    # cv2.waitKey(0)
    return img_array[h_min:h_max, w_min:w_max]


original_folder = "./images-20221018T171551Z-001/images/"
result_folder = "./test"

images = os.listdir(original_folder)[:1000]

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

for image in images:
    image_path = os.path.join(original_folder, image)
    img = cv2.imread(image_path)
    crop_img = text_localization(img)
    result_path = os.path.join(result_folder, image)
    cv2.imwrite(result_path, crop_img)

import cv2

from augment import *


def background_preprocess(img_array):
    brown_lo = np.array([0])
    brown_hi = np.array([200])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, brown_lo, brown_hi)
    h, w = np.where(mask <= 0)

    new_image = img_array.copy()
    new_image[h, w] = (255, 255, 255)
    return new_image


image_path = "./background/vetmuc.png"
img = cv2.imread(image_path)
new_img = background_preprocess(img)
cv2.imwrite("./background/vetmuc_background.jpg", new_img)

import cv2
from matplotlib import pyplot as plt
import numpy as np
import random


def text_localization(img_array):
    width, height = img_array.shape[1], img_array.shape[0]
    brown_lo = np.array([0])
    # brown_hi = np.array([img_array.max() - 90])
    brown_hi = np.array([150])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    hsv = cv2.GaussianBlur(hsv, (9, 9), 0)
    hsv = cv2.adaptiveThreshold(
        hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)
    mask = cv2.inRange(hsv, brown_lo, brown_hi)
    h, w = np.where(mask > 0)

    cond = (h < height - 20) & (w < width - 20) & (h > 20) & (w > 20)
    h_min = np.min(h[cond])
    h_max = np.max(h[cond])

    w_min = np.min(w[cond])
    w_max = np.max(w[cond])

    color = img_array[h[0], w[0]]
    color = (int(color[0]), int(color[1]), int(color[2]))

    return h_min, h_max, w_min, w_max, color


def blur_filter(img_array):
    return cv2.blur(img_array, (8, 8))


def random_stretch(img_array):
    stretch = (random.random() - 0.5)  # -0.5 .. +0.5
    # random width, but at least 1
    wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
    # stretch horizontally by factor 0.5 .. 1.5
    img = cv2.resize(img, (wStretched, img.shape[0]))
    return img


def overlay_image(img_array, background):
    width, height = img_array.shape[1], img_array.shape[0]
    dim = (width, height)
    resized_background = cv2.resize(
        background, dim, interpolation=cv2.INTER_AREA)
    added_image = cv2.addWeighted(img_array, 0.5, resized_background, 0.9, 0)
    return added_image


def change_text_color(img_array):
    img_array = img_array.copy()

    brown_lo = np.array([0])
    brown_hi = np.array([img_array.max() - 100])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, brown_lo, brown_hi)
    img_array[mask > 0] = (0, 0, 255)
    return img_array


def distort(img, orientation='horizontal', func=np.sin, x_scale=0.05, y_scale=5):
    assert orientation[:3] in [
        'hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
    assert func in [
        np.sin, np.cos], "supported functions are np.sin and np.cos"
    assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(
        img.shape[0], img.shape[1]), "y_scale should be less then image size"
    img_dist = img.copy()

    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))

    for c in range(3):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i, c] = np.roll(img[:, i, c], shift(i))
            else:
                img_dist[i, :, c] = np.roll(img[i, :, c], shift(i))

    return img_dist


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1
    noise[mask_overflow_lower] = 0
    img += noise.astype('uint8')
    return img


def text_delete_line(img_array):
    img_array = img_array.copy()
    h_min, h_max, w_min, w_max, color = text_localization(img_array)

    h_range = range(h_min, h_max+1)
    w_range = range(w_min, w_max+1)

    h_choice = random.choices(h_range, k=2)

    start_point = (w_min, h_choice[0] + 10)
    end_point = (w_max, h_choice[1])

    thickness = random.randint(2, 4)
    return cv2.line(img_array,
                    start_point, end_point,
                    color, thickness)


def underline_text(img_array):
    img_array = img_array.copy()
    h_min, h_max, w_min, w_max, color = text_localization(img_array)

    h_range = np.arange(1, 20).tolist()
    h_choice = random.choices(h_range, k=2)

    start_point = (w_min, h_max + h_choice[0])
    end_point = (w_max, h_max + h_choice[1])

    thickness = random.randint(2, 4)
    return cv2.line(img_array,
                    start_point, end_point,
                    color, thickness)


def crop_image(img_array):
    # h_min, h_max, w_min, w_max, _ = text_localization(img_array)
    # return img_array[h_min:h_max, w_min:w_max]
    brown_lo = np.array([0, 0, 0])
    brown_hi = np.array([150, 150, 150])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, brown_lo, brown_hi)
    h, w = np.where(mask > 0)

    h_min = np.min(h)
    h_max = np.max(h)

    w_min = np.min(w)
    w_max = np.max(w)
    return img_array[h_min:h_max, w_min:w_max]

import os

import random
import numpy as np
import cv2

background_folder = "./background"
backgrounds = os.listdir(background_folder)

colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def text_localization(img_array):
    width, height = img_array.shape[1], img_array.shape[0]
    brown_lo = np.array([0])
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
    return cv2.blur(img_array, (5, 5))


def random_stretch(img_array):
    stretch = (random.random() - 0.5)
    wStretched = max(int(img_array.shape[1] * (1 + stretch)), 1)
    img_array = cv2.resize(img_array, (wStretched, img_array.shape[0]))
    return img_array


def overlay_image(img_array):
    # shape of original image
    width, height = img_array.shape[1], img_array.shape[0]
    # random choice background from folder
    background_path = os.path.join(
        background_folder, random.choice(backgrounds))
    background = cv2.imread(background_path)

    # create white background
    final_background = np.full((height, width, 3), 255).astype(np.uint8)

    # determine position of background
    h_value = random.sample(range(height), k=2)
    w_value = random.sample(range(width), k=2)

    min_h, max_h = min(h_value), max(h_value)
    min_w, max_w = min(w_value), max(w_value)

    # overlay background to base image
    resized_dim = (max_w - min_w, max_h - min_h)
    resized_background = cv2.resize(
        background, resized_dim, interpolation=cv2.INTER_AREA)

    resized_background = resized_background.astype(np.uint16)
    final_background = final_background.astype(np.uint16)
    final_background[min_h:max_h, min_w:max_w] += resized_background

    final_background[np.where(final_background == 255)] = 0
    final_background[np.where(final_background > 255)] -= 255

    # area not in background will be white
    mask = np.full(shape=(height, width), fill_value=True)
    mask[min_h:max_h, min_w:max_w] = False

    final_background[mask] = (255, 255, 255)

    final_background = final_background.astype(np.uint8)
    added_image = cv2.addWeighted(img_array, 1, final_background, 0.2, 0)
    return added_image


def change_text_color(base_img):
    img_array = base_img.copy()

    brown_lo = np.array([0])
    brown_hi = np.array([15])

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    hsv = cv2.GaussianBlur(hsv, (9, 9), 0)
    hsv = cv2.adaptiveThreshold(
        hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    color = random.choice(colors)
    img_array[mask > 0] = color
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

    h_choice = random.choices(h_range, k=2)

    start_point = (w_min, h_choice[0] + 10)
    end_point = (w_max, h_choice[1])

    thickness = random.randint(2, 4)
    return cv2.line(img_array,
                    start_point, end_point,
                    color, thickness)


def underline_text(img_array):
    img_array = img_array.copy()
    _, h_max, w_min, w_max, color = text_localization(img_array)

    h_range = np.arange(1, 5).tolist()
    h_choice = random.choices(h_range, k=2)

    start_point = (w_min, h_max + h_choice[0])
    end_point = (w_max, h_max + h_choice[1])

    thickness = random.randint(2, 4)
    return cv2.line(img_array,
                    start_point, end_point,
                    color, thickness)


def crop_image(img_array):
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


augment_function_map = {
    "color": change_text_color,
    "underline": underline_text,
    "delete": text_delete_line,
    "blur": blur_filter,
    "stretch": random_stretch,
    "distort": distort,
    "noise": gaussian_noise,
    "bg": overlay_image
}

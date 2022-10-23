import os
import argparse

import cv2

from augment import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--origin", default="./images-20221018T171551Z-001/images/")
parser.add_argument("--result", default="./result")
parser.add_argument("--gen", default=1)

args = parser.parse_args()

original_folder = args.origin
result_folder = args.result
n_image = int(args.gen)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

augment_function_list = [change_text_color, underline_text, text_delete_line, blur_filter,
                         random_stretch, distort, gaussian_noise, overlay_image]

augment_prob = [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


def gen_image(image, augment_prob, augment_function_list, n_image,
              base_name, result_folder):

    def augment_ocr(image):
        n_function = len(augment_function_list)
        noise = image.copy()
        for i in range(n_function):
            prob_function = augment_prob[i]
            is_apply_function = np.random.choice(
                [0, 1], size=1, p=[1-prob_function, prob_function])
            if is_apply_function == 1:
                noise = augment_function_list[i](noise)
        return noise

    for i in range(n_image):
        image_path = os.path.join(result_folder, f"{base_name}_{i}.jpg")
        augment_image = augment_ocr(image)
        cv2.imwrite(image_path, augment_image)


for image_file in os.listdir(original_folder):
    image_path = os.path.join(original_folder, image_file)
    img = cv2.imread(image_path)
    gen_image(img, augment_prob, augment_function_list,
              n_image=n_image, base_name=image_file,
              result_folder=result_folder)

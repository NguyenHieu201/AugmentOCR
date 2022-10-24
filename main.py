import os
import argparse

import cv2

from augment import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--origin", default="./images-20221018T171551Z-001/images/")
parser.add_argument("--result", default="./result")
parser.add_argument("--gen", default=1)
parser.add_argument("--method", default="all", nargs="+", type=str)
parser.add_argument("--prob", default="-1", nargs="+", type=float)

args = parser.parse_args()

original_folder = args.origin
result_folder = args.result
n_image = int(args.gen)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

augment_function_list = args.method
augment_prob = args.prob

if augment_function_list == "all":
    augment_function_list = ["color", "underline",
                             "delete", "blur", "stretch", "distort", "noise", "bg"]

if augment_prob == -1:
    augment_prob = [1] * len(augment_function_list)


def gen_image(image, augment_prob, augment_function_list, n_image,
              base_name, result_folder):

    def augment_ocr(image):
        extend_name = ""
        n_function = len(augment_function_list)
        noise = image.copy()
        for i in range(n_function):
            augment_name = augment_function_list[i]
            prob_function = augment_prob[i]
            augment_function = augment_function_map[augment_name]
            is_apply_function = np.random.choice(
                [0, 1], size=1, p=[1-prob_function, prob_function])
            if is_apply_function == 1:
                noise = augment_function(noise)
                extend_name += (augment_name + "_")
        return noise, extend_name

    for i in range(n_image):
        augment_image, extend_name = augment_ocr(image)
        image_path = os.path.join(
            result_folder, f"{base_name}_{extend_name}_{i}.jpg")
        cv2.imwrite(image_path, augment_image)


for image_file in os.listdir(original_folder)[:100]:
    image_path = os.path.join(original_folder, image_file)
    img = cv2.imread(image_path)
    gen_image(img, augment_prob, augment_function_list,
              n_image=n_image, base_name=image_file,
              result_folder=result_folder)

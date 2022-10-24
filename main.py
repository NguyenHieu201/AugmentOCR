import math
from multiprocessing import Process
from augment import *
import cv2
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--origin", default="./images-20221018T171551Z-001/images/")
parser.add_argument("--result", default="./result")
parser.add_argument("--gen", default=1, type=int)
parser.add_argument("--method", default="all", nargs="+", type=str)
parser.add_argument("--prob", default="-1", nargs="+", type=float)
parser.add_argument("--batch", default="10", type=int)


def gen_image(image, augment_prob, augment_function_list, index_range,
              base_name, result_folder):
    for index in index_range:
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

        image_path = os.path.join(
            result_folder, f"{base_name}_{extend_name}_{index}.jpg")
        cv2.imwrite(image_path, noise)


if __name__ == "__main__":
    args = parser.parse_args()

    original_folder = args.origin
    result_folder = args.result
    gen = args.gen
    batch = args.batch

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    augment_function_list = args.method
    augment_prob = args.prob

    # Process argument input
    if augment_function_list == "all":
        augment_function_list = ["color", "underline",
                                 "delete", "blur", "stretch", "distort", "noise", "bg"]

    if augment_prob == -1:
        augment_prob = [1] * len(augment_function_list)

    if gen < batch:
        batch = gen

    # multi processing
    image_files = os.listdir(original_folder)[:1]

    for image_file in image_files:
        image_path = os.path.join(original_folder, image_file)
        img = cv2.imread(image_path)
        number_of_batch = math.ceil(gen / batch)
        index_ranges = [list(range(batch * i, batch * i + batch))
                        for i in range(number_of_batch - 1)]
        index_ranges += [list(range(batch * (number_of_batch - 1), gen))]

        processes = [Process(target=gen_image,
                             args=(img, augment_prob, augment_function_list,
                                   index_ranges[i], image_file,
                                   result_folder)) for i in range(len(index_ranges))]

        [p.start() for p in processes]

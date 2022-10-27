import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import contours
from scipy import ndimage
from tqdm import tqdm


def detect_paper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10  # Max allowed gap between line for joining them
    )

    for points in lines:
        x1, y1, x2, y2 = points[0]
        lines_list += [(x1, y1), (x2, y2)]

    lines_coordinate = np.array(lines_list)
    min_h, min_w = np.min(lines_coordinate, axis=0)
    max_h, max_w = np.max(lines_coordinate, axis=0)

    only_paper_image = image[min_w:max_w, min_h:max_h]
    return only_paper_image


def extract_box(image):
    # ref: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    # rotation angle in degree
    image = ndimage.rotate(image, -1)

    # ref: https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv
    # Load image, grayscale, and adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # Filter out all words and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (150, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              vertical_kernel, iterations=9)

    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    # Loop over the boxes
    boxes = []
    show_image = image.copy()
    for c in cnts:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(c)

        # Ignore the small boxes
        if w < 200 or h < 100:
            continue
        if w > 500 or h > 500:
            continue
        if w / h > 3:
            continue
        # Append the box to the list
        boxes.append([x, y, w, h])

    # get all image boxes
    result = []

    test_image = image.copy()
    for box in boxes:
        x, y, w, h = box
        result.append(image[y:y+h, x:x+w])

    #     start_point = (x, y)
    #     end_point = (x+w, y+h)

    #     color = (255, 0, 0)
    #     thickness = 2

    #     test_image = cv2.rectangle(test_image, start_point, end_point,
    #                                color, thickness)
    #     cv2.rectangle(show_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    # plt.imshow(test_image)
    # plt.show()
    return show_image, result


def preprocessing(img):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def combine_extract_box(image):
    image_list = []
    img = image.copy()
    # img = detect_paper(image)
    a, b = extract_box(img)
    for i in range(9):
        for j in range(8):
            image_list.append(b[i*8+j])

    return image_list


original_folder = "./only_paper"
result_folder = "./only_text"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

images = os.listdir(original_folder)[:1]
error_list = []
# for image in tqdm(images):
for image in images:
    i = 0
    image_path = os.path.join(original_folder, image)
    img = cv2.imread(image_path)

    # Image fit to paper
    # height, width, _ = img.shape
    # width_bound, height_bound = 450, 100
    # img = img[height_bound:height-height_bound, width_bound:width-width_bound]
    # cv2.imwrite(os.path.join(result_folder, f"{image}.jpg"), img)

    # Cut box
    try:
        text_image_list = combine_extract_box(img)
        for text_image in text_image_list:
            cv2.imwrite(os.path.join(result_folder,
                        f"{image}_{i}.jpg"), text_image)
            i += 1
    except IndexError:
        error_list.append(image)
print(len(error_list))
[print(error_image) for error_image in error_list]

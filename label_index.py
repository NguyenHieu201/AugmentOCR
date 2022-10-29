import os

from threading import Thread
import cv2

shouldExit = False
color = True

origin_path = "./new_origin"
images = os.listdir(origin_path)
index_image = [""] * len(images)


def cam(img):
    while not shouldExit:
        cv2.imshow("window title", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


for image in images:
    shouldExit = False
    image_path = os.path.join(origin_path, image)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1200, 800))
    t = Thread(target=cam, args=(img,))
    t.start()
    print('select an action: q-quit, t-toggle color')
    choice = input(">").strip().lower()
    if choice == 'q':
        shouldExit = True
        t.join()
    else:
        print('invalid input')

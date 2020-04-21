# def get_crop(img, pct):
#     x = int(img.shape[0] * ((1 - pct) / 2))
#     h = int(img.shape[0] * pct)
#     y = int(img.shape[1] * ((1 - pct) / 2))
#     w = int(img.shape[1] * pct)
#     img = img[x:x + h, y:y + w]
#     return img


def show_images(path, feat, pcts):
    import cv2
    import matplotlib.pyplot as plt
    from skimage.filters import laplace, sobel, roberts

    for pct in pcts:
        plt.figure(figsize=(10, 10))
        x_label = ['Sharp', 'Defocused Blur', 'Motion Blur']
        for i in range(len(path)):
            if feat == None:
                img = cv2.imread(str(path[i]))  # read image in BGR
                img = get_crop(img, pct)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(str(path[i]), 0)  # read gray scale image
                img = get_crop(img, pct)

            if feat == 'laplace':
                img = laplace(img)
            elif feat == 'sobel':
                img = sobel(img)
            elif feat == 'roberts':
                img = roberts(img)
            elif feat == 'canny':
                img = cv2.Canny(img, 100, 200, 3, L2gradient=True)
            elif feat == None:
                pass
            else:
                raise ValueError('Feature must be None, laplace, sobel or roberts')

            plt.subplot(1, 3, i + 1)
            plt.imshow(img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            xlabel_temp = f'{x_label[i]} ({pct:.0%} crop)'
            # plt.xlabel(x_label[i]+' '+str(pct))
            plt.xlabel(xlabel_temp)

        plt.tight_layout()
        plt.show()
    # print(img.shape)


def get_data(images, pct):
    import os
    import concurrent.futures
    from functools import partial
    import pandas as pd
    # images = [os.path.join(path, image) for image in images]
    func = partial(process_img, pct)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        features = executor.map(func, images)
    df = pd.DataFrame(features)
    df.drop(0, axis=1, inplace=True)
    return df


def process_img(pct, img):
    import cv2
    import numpy as np
    from cv2 import Canny
    from skimage.filters import laplace, sobel, roberts
    feature = []

    image_gray = cv2.imread(str(img), 0)

    image_gray = get_crop(image_gray, pct)

    lap_feat = laplace(image_gray)
    sob_feat = sobel(image_gray)
    rob_feat = roberts(image_gray)
    can_feat = Canny(image_gray, 100, 200, 3, L2gradient=True)
    feature.extend([img, lap_feat.mean(), lap_feat.var(), np.amax(lap_feat),
                    sob_feat.mean(), sob_feat.var(), np.max(sob_feat),
                    rob_feat.mean(), rob_feat.var(), np.max(rob_feat),
                    can_feat.mean(), can_feat.var(), np.max(can_feat)])
    return feature


def save_faces(img):
    import os
    import cv2
    img2 = img.replace(r'\03_sharp_photos', r'\04_faces\04_faces_harr')
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        _, file_extension = os.path.splitext(os.path.basename(img2))
        cv2.imwrite(img2[:-len(file_extension)] + '_w_' + str(w) + '_h_' + str(h) + '_faces.jpg', roi_color)
    return True


def find_faces(images):
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(save_faces, images)


def convert_files(files):
    import sys
    from pathlib import Path
    import imageio
    from utility import read_image
    from PIL import Image
    import cv2

    for s_file, t_file in files:
        if not t_file.is_file():
            _, img, _ = read_image(s_file, resize_img=False)
            if s_file.suffix != '.CR2':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imageio.imsave(t_file, img)
    return
    # file_base, file_extension = os.path.splitext(os.path.basename(source))


def move_classified_results(images_results):
    import os
    import shutil
    for img, res in images_results:
        # img, res = zip(*soccer_images_results)
        if res == 0:  # image is blurred
            img2 = img.replace(r'\02_converted_photos', r'\03_blurry_photos')
            if not os.path.isfile(img2):
                shutil.copy(img, img2)
        elif res == 1:  # image is sharp
            img2 = img.replace(r'\02_converted_photos', r'\03_sharp_photos')
            if not os.path.isfile(img2):
                shutil.copy(img, img2)
    return True

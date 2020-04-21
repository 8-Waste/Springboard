import settings as s

import rawpy
import cv2
from pathlib import Path
import os


def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def read_image(file, resize_img = True):
    import PIL
    if file.suffix.lower() == '.cr2':
        with rawpy.imread(str(file)) as raw:
            image = raw.postprocess()
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        file = Path(file.with_suffix('.tiff'))
    elif file.suffix.lower() in ['.jpg', 'jpeg']:
        image = cv2.imread(str(file))
    elif file.suffix.lower() in ['.tiff', '.tif']:
        image = cv2.imread(str(file))

    else:
        image = None
        print('file type not found: ' + str(file.suffix))
        return False, image, 0
    pct_resize = 1
    if resize_img:
        image, pct_resize = image_resize(image, height=1350)
    return True, image, pct_resize

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image and the ratio used
    return resized, r

import settings as s
from pathlib import Path
import cv2
import pickle
import random
from utility import read_image
import pprint
from time import perf_counter

pp = pprint.PrettyPrinter(depth=6)

def tag_images(skip_completed=True, k=-1):
    assert Path.cwd().name == 'code'
    random.seed(s.random_seed)

    def onMouse(event, x, y, flags, param):
        nonlocal sc_list, pct_resize, image
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 25, (0, 155, 255), 2)
            cv2.imshow('image', image)
            face_nbr = str(len(sc_list) + 1).zfill(3)
            # store values based on 100% image size
            entry = {'face_gt_nbr': face_nbr, 'x': int(x / pct_resize), 'y': int(y / pct_resize)}
            sc_list.append(entry)
        elif event == cv2.EVENT_RBUTTONDOWN:
            sc_list = []
            ret, image, pct_resize = read_image(file)
            cv2.imshow('image', image)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)


    # retrieve all the cr2 and jpg files from the source directory  #TODO consider adding tiff files
    # files = [p.resolve() for p in Path(s.SOURCE_DIR).rglob("*") if
    #          p.suffix.lower() in [".cr2", ".jpg"]]  # includes all subdirectories

    files = s.SOURCE_JPGS_LIST
    print(len(files))

    if k > 0: # if k is negative or 0, include all files by skipping line
        files = random.choices(population=files, k=k)

    for file in files:
        sc_list = []  # gt = ground truth
        sc_file = Path(file.with_suffix('.pkl'))  # always same directory & filestem
        if sc_file.is_file() and skip_completed:
            continue
        ret, image, pct_resize = read_image(file, True)
        if not ret:
            raise RuntimeError("Error opening " + str(file))

        # if there are existing faces marked, retrieve and show on image
        if sc_file.is_file():
            with (open(sc_file, "rb")) as openfile:
                sc_list = pickle.load(openfile)
                sc_list = sc_list['ground_truth']
            for i, gt in enumerate(sc_list):
                cv2.circle(image, (int(gt['x'] * pct_resize), int(gt['y'] * pct_resize)),
                           25, (0, 155, 255), 2)
        start_time = perf_counter()
        print(file)
        while True:
            cv2.imshow('image', image)
            k = cv2.waitKey(1)
            if k == 32:  # space pressed  - go to next image
                break
            if k == 27:  # esc pressed = exit this and upper loop
                cv2.destroyAllWindows()
                break
        if k == 27:  # esc was pressed
            break

        elapsed_time = round(perf_counter() - start_time, 2)
        sc_dict = {'ground_truth': sc_list}
        sc_dict['original_image_info'] = {'h': image.shape[0], 'w': image.shape[1],
                                             'sq_pixels': image.shape[0] * image.shape[1], 'filename': str(file.name),
                                             'filetype': str(file.suffix),
                                             'elapsed_time': elapsed_time}  # TODO need to consider adding source

        with open(str(sc_file), 'wb') as handle:
            pickle.dump(sc_dict, handle)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    tag_images(skip_completed=True, k=-1)

import settings as s
from pathlib import Path
import cv2
import rawpy
import os
import numpy as np
import pickle
from utility import read_image, image_resize
from data_storytelling_support import dominate_colors


def show_errors_gt_ff():
    assert Path.cwd().name == 'code'

    files = [p.resolve() for p in Path(s.output).glob("**/*") if p.suffix.lower() in [".jpg"] if p.parent.name != 'faces' if not str(p.stem).endswith('dominatecolors')]
    for file in files:
        ret, image, pct_resize = read_image(file, True)
        if not ret:
            assert 1 == 2
        posList = []
        sidecar_file = Path(file.with_suffix('.pkl'))
        if sidecar_file.is_file():
            with (open(sidecar_file, "rb")) as openfile:
                posList = pickle.load(openfile)
        else:
            assert 1 == 2

        for pos in posList['ground_truth']:
            if not pos['mtcnn_found']:
                cv2.circle(image, (int(pos['x'] * pct_resize), int(pos['y'] * pct_resize)), 25,
                           (102, 255, 0), 2)
            else:
                cv2.circle(image, (int(pos['x'] * pct_resize), int(pos['y'] * pct_resize)), 25,
                           (0, 155, 255), 2)

        for each in posList['algorithm_faces']:
            bounding_box = each['box']
            if not each['confirmed']:
                cv2.rectangle(image,
                              (int(bounding_box[0] * pct_resize), int(bounding_box[1] * pct_resize)),
                              (int((bounding_box[0] + bounding_box[2]) * pct_resize), int((bounding_box[1] + bounding_box[3])* pct_resize)),
                              (102, 255, 0),
                              2)
            else:
                cv2.rectangle(image,
                              (int(bounding_box[0] * pct_resize), int(bounding_box[1] * pct_resize)),
                              (int((bounding_box[0] + bounding_box[2]) * pct_resize), int((bounding_box[1] + bounding_box[3])* pct_resize)),
                              (0, 155, 255),
                              2)

        while True:
            cv2.imshow('image', image)
            k = cv2.waitKey(1)
            if k == 32:
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_errors_gt_ff()
import settings as s
import cv2
import sys
from mtcnn import MTCNN
from pathlib import Path
import pickle
from utility import read_image
import pprint
from time import perf_counter
pp = pprint.PrettyPrinter(depth=6)


def find_faces_in_image(new_file, image_resized, faces_export_dir, img):
    detector = MTCNN()  # we are finding this on the scaled image and storing points based on the scaled image
    result = detector.detect_faces(image_resized)
    for i, each in enumerate(result):
        result[i]['confirmed'] = False
     # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    for i in range(len(result)):
        bounding_box = result[i]['box']
        face = image_resized[bounding_box[1]:(bounding_box[1] + bounding_box[3]),
               bounding_box[0]:(bounding_box[0] + bounding_box[2])]
        face_file = Path(faces_export_dir, new_file.stem + '_faceno_' + str(i + 1).zfill(3) + new_file.suffix)
        try:
            cv2.imwrite(str(face_file), face)
        except:
            print(f'missed {new_file.stem}{new_file.suffix}')  # TODO need to change this to a raise error
        pickle_file = Path(faces_export_dir, new_file.stem + '_faceno_' + str(i + 1).zfill(3) + '.pkl')
        image_info = {'org_image_info': {'h': img.shape[0], 'w': img.shape[1], 'sq_pixels': img.shape[0] * img.shape[1]}}  # TODO need to consider adding source

        with open(pickle_file, 'wb') as handle:
            pickle.dump(image_info, handle)
    return result


def find_faces():
    assert Path.cwd().name == 'code'

    # pct_of_orgs = np.linspace(25,100,4, endpoint=True, dtype=int)
    pct_of_orgs = [100]
    files = [p.resolve() for p in Path(s.source).glob("**/*") if p.suffix.lower() in [".cr2", ".jpg"]]
    files = [file for file in files if file.with_suffix('.pkl').is_file()]

    for i, file in enumerate(files):
        print(str(i + 1) + ' of ' + str(len(files)))
        export_dir = s.output.resolve() / file.parent.relative_to(s.source.resolve())  # export directory
        faces_export_dir = Path(str(export_dir), 'faces')
        faces_export_dir.mkdir(parents=True, exist_ok=True)  # create export directory if does not exist

        ret, image, pct_resize = read_image(file, False)
        if not ret:
            raise RuntimeError("Source file not opened")
        for pct_of_org in pct_of_orgs:
            start_time = perf_counter()
            # resize the image to the scaling factor and save to output/??
            image_resized = cv2.resize(image, (0, 0), fx=pct_of_org / 100, fy=pct_of_org / 100)
            new_file = Path(export_dir, file.stem + '_' + str(pct_of_org).zfill(3) + '.jpg')
            cv2.imwrite(str(new_file), image_resized)  # save the resized image, file not adjusted for viewing

            # copy sidecar_file_gt (ground truth for faces) from source to output with appropriate pctg
            sidecar_file = file.with_suffix('.pkl')  # source
            with (open(sidecar_file, "rb")) as openfile:
                posList = pickle.load(openfile)
            sidecar_file = Path(export_dir, file.stem + '_' + str(pct_of_org).zfill(3) + '.pkl')  # output
            # update ground_truth x and y based on percentage of shrinking image
            for i, pos in enumerate(posList['ground_truth']):
                pos['x'] = int(pos['x'] * (pct_of_org / 100))
                pos['y'] = int(pos['y'] * (pct_of_org / 100))
                pos['mtcnn_found'] = False
            with open(str(sidecar_file), 'wb') as handle:
                pickle.dump(posList, handle)  # done copying original ground truth info can saving new sidecar file, overrights any previous file
            # TODO need to consider what happen if file already exists

            sidecar_file = Path(new_file.with_suffix('.pkl'))

            result = find_faces_in_image(new_file, image_resized, faces_export_dir, image)
            if sidecar_file.is_file():
                with (open(sidecar_file, "rb")) as openfile:
                    posList = pickle.load(openfile)
            else:
                pass # need to raise an error if no file here.
            posList['algorithm_faces'] = result
            elapsed_time = round(perf_counter() - start_time,2)
            # posList['XXXXXXXX'] = elapsed_time  #TODO this doesn't seem to be working
            with open(str(sidecar_file), 'wb') as handle:
                pickle.dump(posList, handle)

if __name__ == '__main__':
    find_faces()

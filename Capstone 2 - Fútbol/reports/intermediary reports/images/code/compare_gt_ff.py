import settings as s
from pathlib import Path
import pickle

def compare_info():
    assert Path.cwd().name == 'code'

    files = [p.resolve() for p in Path(s.output).rglob("**/*") if p.suffix.lower() in [".jpg"] if p.parent.name != 'faces' if not p.stem.endswith('_dominatecolors')]

    for file in files:
        sidecar_file = file.with_suffix('.pkl')
        if sidecar_file.is_file():
            with (open(sidecar_file, "rb")) as openfile:
                posList = pickle.load(openfile)
                print(posList)
        else:
            assert 1 == 2
        for i1, pos in enumerate(posList['ground_truth']):
            x = int(pos['x'])
            y = int(pos['y'])

            for i2, each in enumerate(posList['algorithm_faces']):
                x1, y1, w, h = each['box']
                x2 = x1 + w
                y2 = y1 + h
                if x1 < x < x2 and y1 < y < y2:
                    pos['mtcnn_found'] = True
                    each['confirmed'] = True
                    break


        faces_result = {}
        faces_result['gt_faces_in_image'] = len(posList['ground_truth'])
        faces_result['gt_faces_in_image_found'] = len([1 for each in posList['ground_truth'] if each['mtcnn_found']==True])
        faces_result['gt_faces_in_image_not_found'] = len([1 for each in posList['ground_truth'] if each['mtcnn_found']==False])
        faces_result['alg_faces_found'] = len(posList['algorithm_faces'])
        faces_result['alg_faces_found_correct'] =  len([1 for each in posList['algorithm_faces'] if each['confirmed'] == True])
        faces_result['alg_faces_found_incorrect'] = len([1 for each in posList['algorithm_faces'] if each['confirmed'] == False])
        # x = {}
        posList['faces_result'] = faces_result
        # print(x)
        # print('')

        # faces_result['Incorrect_faces_found'] = len([1 for each in posList['ground_truth'] if each['mtcnn_found']==False])
        # faces_result['total_faces_found'] = len(posList['algorithm_faces'])
        # faces_result['faces_not_found'] = faces_in_image - correct_faces_found
        # faces_not_found = faces_in_image - correct_faces_found
        #
        # print('-'*50)

        with open(str(sidecar_file), 'wb') as handle:
            pickle.dump(posList, handle)
if __name__ == '__main__':
    compare_info()
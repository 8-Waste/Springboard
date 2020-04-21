import cv2
import rawpy
import pickle
import settings as s
from pathlib import Path
from time import perf_counter

class process_image(object):
    '''Process image from source to output'''

    version = '0.1'                 # class variable (use strings)

    def __init__(self, file_org: Path):
        self.file_org = file_org
        print(type(file_org))
        print(self.version)
        version = '009'

    def return_filename(self):

        return self.file_org

    def read_image(self):
        '''Read image from file'''
        print(f'asdfasdf {self.version}')
        print(f'my file {type(self.file_org)}')
        version = '0.2'
        print(f'asdfasdf {version}')
        if self.file_org.suffix.lower() == '.cr2':
            with rawpy.imread(str(self.file_org)) as raw:
                self.image = raw.postprocess()
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif self.file_org.suffix.lower() in ['.jpg', 'jpeg']:
            self.image = cv2.imread(str(self.file_org))
        else:
            self.image = None
            print('file type not found: ' + str(self.file_org.suffix))
            return False, self.image, 0
        return self.image

    def find_faces(self):
        assert Path.cwd().name == 'code'

        self.pct_of_orgs = [100]

        self.export_dir = s.output.resolve() / self.file_org.parent.relative_to(s.source.resolve())  # export directory
        self.faces_export_dir = Path(str(self.export_dir), 'faces')
        self.faces_export_dir.mkdir(parents=True, exist_ok=True)  # create export directory if does not exist

        for self.pct_of_org in self.pct_of_orgs:
            start_time = perf_counter()
            # resize the image to the scaling factor and save to output/??
            self.image_resized = cv2.resize(self.image, (0, 0), fx=self.pct_of_org / 100, fy=self.pct_of_org / 100)
            self.new_file = Path(self.export_dir, self.file_org.stem + '_' + str(self.pct_of_org).zfill(3) + '.jpg')
            print('Here')
            cv2.imwrite(str(self.new_file), self.image_resized)  # save the resized image, file not adjusted for viewing

            # copy sidecar_file_gt (ground truth for faces) from source to output with appropriate pctg
            self.sidecar_file = self.file_org.with_suffix('.pkl')  # source
            with (open(self.sidecar_file, "rb")) as openfile:
                posList = pickle.load(openfile)
            self.sidecar_file = Path(self.export_dir, self.file_org.stem + '_' + str(self.pct_of_org).zfill(3) + '.pkl')  # output
            # update ground_truth x and y based on percentage of shrinking image
            for i, pos in enumerate(posList['ground_truth']):
                pos['x'] = int(pos['x'] * (self.pct_of_org / 100))
                pos['y'] = int(pos['y'] * (self.pct_of_org / 100))
                pos['mtcnn_found'] = False
            with open(str(self.sidecar_file), 'wb') as handle:
                pickle.dump(posList,
                            handle)  # done copying original ground truth info can saving new sidecar file, overrights any previous file
            # TODO need to consider what happen if file already exists

            # sidecar_file = Path(new_file.with_suffix('.pkl'))
            #
            # result = find_faces_in_image(new_file, image_resized, faces_export_dir, image)
            # if sidecar_file.is_file():
            #     with (open(sidecar_file, "rb")) as openfile:(
            #         posList = pickle.load(openfile)
            # else:
            #     pass  # need to raise an error if no file here.
            # posList['algorithm_faces'] = result
            # elapsed_time = round(perf_counter() - start_time, 2)
            # # posList['XXXXXXXX'] = elapsed_time  #TODO this doesn't seem to be working
            # with open(str(sidecar_file), 'wb') as handle:
            #     pickle.dump(posList, handle)




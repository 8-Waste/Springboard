from pathlib import Path

DATA_DIR = Path('../data/')
DATA_DIR_jn = Path('..', DATA_DIR)

MODELS = Path(DATA_DIR / '00_support/models/')
MODELS_jn = Path('..', MODELS)

SOURCE = Path(DATA_DIR / '01_source_photos/')
SOURCE_jn = Path('..', SOURCE)

TARGET_FACES = Path(DATA_DIR / '02_target_faces/')
TARGET_FACES_jn = Path('..', TARGET_FACES)

UNKNOWN_FACES = Path(DATA_DIR / '03_unknown_faces/')
UNKNOWN_FACES_jn = Path('..' / UNKNOWN_FACES)

RECOGNIZE_RESULTS = Path(DATA_DIR / '04_found_targets/')
RECOGNIZE_RESULTS_jn = Path('..', RECOGNIZE_RESULTS)





FOCUSED_VS_UNFOCUSED_DATASET = Path(DATA_DIR / '00_support/focused_vs_unfocused_dataset/')
FOCUSED_VS_UNFOCUSED_DATASET_jn = Path('..' / FOCUSED_VS_UNFOCUSED_DATASET)

FACE_VS_OBJECT_DATASET = Path(DATA_DIR / '00_support/face_vs_object_dataset/')
FACE_VS_OBJECT_DATASET_jn = Path('..' / FACE_VS_OBJECT_DATASET)


# FOCUSED_FACES_DATASET_DIR = Path(DATA_DIR / '00_other/blur_datasets/faces_blur_dataset')
# FOCUSED_FACES_DATASET_DIR_jn = Path('..' / FOCUSED_FACES_DATASET_DIR)
#
REPORT_IMAGES = Path(DATA_DIR.parent / 'reports/intermediary reports/images')
REPORT_IMAGES_jn = Path('..' / REPORT_IMAGES)
#
# SOURCE_JPGS_LIST = [f_name for d_name in Path(SOURCE_DIR).glob("*") for f_name in d_name.glob('*') if d_name.is_dir() if f_name.is_file() if f_name.suffix=='.jpg']
#
# OTHER_UNKNOWN = [f_name for d_name in Path('/target_photos/').glob("*") for f_name in d_name.glob('*') if d_name.is_dir() if f_name.is_file() if f_name.suffix=='.jpg' if str(d_name) == '11_Elijah Widdows']

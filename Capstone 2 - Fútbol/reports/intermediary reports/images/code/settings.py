from pathlib import Path

random_seed = 7

DATA_DIR = Path('../data/')
DATA_DIR_jn = Path('..', DATA_DIR)

MODELS_DIR = Path(DATA_DIR / '10_other/models/')
MODELS_DIR_jn = Path('..', MODELS_DIR)

OUTPUT_DIR = Path(DATA_DIR / 'output/')
OUTPUT_DIR_jn = Path('..', OUTPUT_DIR)

SOURCE_DIR = Path(DATA_DIR / '01_source_photos/')
SOURCE_DIR_jn = Path('..', SOURCE_DIR)

TARGET_DIR = Path(DATA_DIR / '02_target_faces/')
TARGET_DIR_jn = Path('..', TARGET_DIR)

UNKNOWN_FACES_DIR = Path(DATA_DIR / '02_unknown_faces/')
UNKNOWN_FACES_DIR_jn = Path('..' / UNKNOWN_FACES_DIR)

FACES_DATASET_DIR = Path(DATA_DIR / '10_other/faces_dataset/')
FACES_DATASET_DIR_jn = Path('..' / FACES_DATASET_DIR)

FOCUSED_FACES_DATASET_DIR = Path(DATA_DIR / '10_other/blur_datasets/faces_blur_dataset')
FOCUSED_FACES_DATASET_DIR_jn = Path('..' / FOCUSED_FACES_DATASET_DIR)

REPORT_IMAGES_DIR = Path(DATA_DIR.parent / 'reports/intermediary reports/images')
REPORT_IMAGES_DIR_jn = Path('..' / REPORT_IMAGES_DIR)

SOURCE_JPGS_LIST = [f_name for d_name in Path(SOURCE_DIR).glob("*") for f_name in d_name.glob('*') if d_name.is_dir() if f_name.is_file() if f_name.suffix=='.jpg']

OTHER_UNKNOWN = [f_name for d_name in Path('/target_photos/').glob("*") for f_name in d_name.glob('*') if d_name.is_dir() if f_name.is_file() if f_name.suffix=='.jpg' if str(d_name) == '11_Elijah Widdows']

# V:\python\springboard\Springboard_Capstone_2_Futbol\reports\intermediary reports\Images
# SOURCE_PATH = Path(DATA_DIR / '01_source_photos/')
# SOURCE_PATH_jn = Path('..', SOURCE_PATH)

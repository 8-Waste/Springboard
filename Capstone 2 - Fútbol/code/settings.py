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

REPORT_IMAGES = Path(DATA_DIR.parent / 'reports/intermediary reports/images')
REPORT_IMAGES_jn = Path('..' / REPORT_IMAGES)

# this dataset is not longer used but the reference remains so the 02 - Data Wrangling notebook will still run
SHARP_VS_BLURRY_DATASET = Path(DATA_DIR / '00_support/sharp_vs_blurry_dataset/')
SHARP_VS_BLURRY_DATASET_jn = Path('..' / SHARP_VS_BLURRY_DATASET)

OLD_MODELS = Path(DATA_DIR / '00_support/models/old_models')
OLD_MODELS_jn = Path('..' / OLD_MODELS)

OLD_SOURCE = Path(DATA_DIR / '01_source_photos/pre_baseline')
OLD_SOURCE_jn = Path('..', OLD_SOURCE)
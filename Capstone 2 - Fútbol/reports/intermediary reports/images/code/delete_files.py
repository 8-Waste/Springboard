import settings as s
from pathlib import Path
import os, shutil

def delete_files(source_pickles=False):
    if source_pickles:
        files = [p.resolve() for p in Path(s.source).glob("**/*") if p.suffix.lower() in [".pkl"]]
        for file in files:
            file.unlink()

    for filename in os.listdir(s.output):
        file_path = os.path.join(s.output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return source_pickles


import settings as s
from process_image import process_image
from pathlib import Path
import cv2

files = [p.resolve() for p in Path(s.source).glob("**/*") if p.suffix.lower() in [".cr2", ".jpg"]]
file = files[0]
print(file)

my_image = process_image(file)
print(my_image.return_filename())
my_image.read_image()
my_image.find_faces()

print(process_image.version)
print(my_image.version)
print(version)

print(type(my_image.file_org))

# cv2.imshow('x', x)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()
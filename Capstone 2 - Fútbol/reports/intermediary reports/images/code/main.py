import settings as s
import os
import time
from label_images import tag_images
from compare_gt_ff import compare_info
from data_storytelling_support import dominate_colors
from delete_files import delete_files
from find_faces import find_faces
from show_errrors import show_errors_gt_ff

os.environ['HDF5_DISABLE_VERSION_CHECK']='1'

def second_menu():
    print("This is the second menu")

def main():
    while True:
        print('\n'*10)
        print("1) Tag images with correct (ground truth) faces:")
        print("2) Have the computer find faces:")
        print("3) Compare ground truth faces with computer found faces")
        print("4) Show errors ground truth faces and computer found faces:")
        print("5) Calculate dominate colors in faces:")
        print("6) Calculate dominate colors in images:")
        print("7) Do something:")
        print("8) Delete all but source pkl and run all:")
        print("9) Exit\n")

        choice = input ("Please make a choice: ")

        if choice == "1":
            tag_images(skip_completed=False)
        elif choice == "2":
            compare_info()
        elif choice == "3":
            print("Do Something 3")
        elif choice == "5":
            dominate_colors(image_type='faces', k=-1)
        elif choice == "6":
            dominate_colors(image_type='images', k=-1)
        elif choice == "8":
            source_pickles = False
            delete_files(source_pickles=source_pickles)
            if source_pickles:
                tag_images(skip_completed=False)
            find_faces()
            compare_info()
            dominate_colors(image_type = 'faces', k = -1)
            dominate_colors(image_type='images', k=-1)
            show_errors_gt_ff()

        elif choice == "9":
            break
        else:
            print("I don't understand your choice.")

if __name__ == '__main__':
    main()
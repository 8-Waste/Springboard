# capstone_2_project
This repository contains the code and reports for my Capstone 2 Project, **Fútbol Match Highlights** for the Springboard Data Science Career Track program.  

## About the project
> “It is not just about the money, it is about what you achieve on the pitch. “ Ronaldinho
 
College coaches require soccer players to submit highlight video and game film to be considered for an athletic scholarship. The issues facing the student-athlete are:
* Recording and processing video is time-consuming
* The athlete is playing so they must rely on others to take the video
* People would rather watch the game instead of being a videographer
* Processing the video is time consuming
* The required hardware and software are expensive.

My Capstone 2 project will focus on identifying a targeted player using facial recognition. I will initially focus on identifying the target player in photos and post project transition to video and teammates.

## The dataset
The dataset is self generated from high school soccer matches.  I used one dataset 
to supplement my Faces vs Objects (non-faces) model.  The dataset can be downlaoded [here](http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/).

## Folder content
**root:** The DataStory.

**data:** Contains all the data for this project.
* **data/00_support:** Contains face_vs_object dataset, focused_vs_unfocused_dataset and all models.
* **data/00_support:** Contains face_vs_object dataset, focused_vs_unfocused_dataset and all models.
* **data/01_source_photos:** Contains photos of players on the pitch.
* **data/02_target_faces:** Contains photos of target player (player's face to locate).
* **data/03_unknown_faces:** Contains photos of faces of anyone except the target player.
* **data/04_found_faces:** Contains photos of recognized images of the target player found in the 01_source_photos.

**code:** Contains the code used for this project excluding Jupyter Notebooks which are located in reports.  The .py files in this directory support Jupyter Notebook via importing.  The allows the Jupyter Notebook to be more more readable.
* **code/mtcnn:** Contains MTCNN detector code.

**reports:** Contains all reports, graphs, and slides for this project.
* **reports/intermediary reports/images:** Contains images used for Jupyter Notebooks and reports
* **reports/intermediary reports:** Contains additional reports regarding project

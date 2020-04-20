# capstone_1_project
This repository contains the code and reports for my Capstone 1 Project, ["Bosch Production Line Performance"](https://www.kaggle.com/c/bosch-production-line-performance) for the Springboard Data Science Career Track program.  The data can be downloaded [here](https://www.kaggle.com/c/bosch-production-line-performance/data).

## About the project
> “An ounce of prevention is worth a pound of cure. “ Benjamin Franklin (1736)
 
Avoiding problems is generally preferable to dealing with their consequences.  Although this concept is usually associated with healthcare, it can also be applied to manufacturing.  Producing defective products (the problems) increases costs, decreases profit, and can cause a customer to reevaluate purchasing your product (the consequences).  Bosch, one of the world’s leading manufacturing companies, has an imperative to reduce defects while ensuring their products are of the highest quality standards.  This project utilizes advanced machine learning techniques combined with Bosch’s detailed manufacturing records to identify defective products.  Bosch will utilize the results to identify the root cause of the defects and implement appropriate corrections.
## The dataset
The dataset represents measurements from 2,367,495 products (rows) as they move through Bosch’s production line.  There are 968 numerical features and 1,156 date features (columns).  The dataset can be downloaded  [here](https://www.kaggle.com/c/bosch-production-line-performance/data). 

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

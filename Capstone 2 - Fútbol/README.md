# capstone_1_project
This repository contains the code and reports for my Capstone 1 Project, ["Bosch Production Line Performance"](https://www.kaggle.com/c/bosch-production-line-performance) for the Springboard Data Science Career Track program.  The data can be downloaded [here](https://www.kaggle.com/c/bosch-production-line-performance/data).

## About the project
> “An ounce of prevention is worth a pound of cure. “ Benjamin Franklin (1736)
 
Avoiding problems is generally preferable to dealing with their consequences.  Although this concept is usually associated with healthcare, it can also be applied to manufacturing.  Producing defective products (the problems) increases costs, decreases profit, and can cause a customer to reevaluate purchasing your product (the consequences).  Bosch, one of the world’s leading manufacturing companies, has an imperative to reduce defects while ensuring their products are of the highest quality standards.  This project utilizes advanced machine learning techniques combined with Bosch’s detailed manufacturing records to identify defective products.  Bosch will utilize the results to identify the root cause of the defects and implement appropriate corrections.
## The dataset
The dataset represents measurements from 2,367,495 products (rows) as they move through Bosch’s production line.  There are 968 numerical features and 1,156 date features (columns).  The dataset can be downloaded  [here](https://www.kaggle.com/c/bosch-production-line-performance/data). 

## Folder content
**root:** The DataStory jupyter notebook.

**data:** The data for this project exceeds 14GB and can be downloaded [here](https://www.kaggle.com/c/bosch-production-line-performance/data). 
* **data/special:** Contains support files for the DataStory jupyter notebook.  The notebook will not execute completely without downloading the data.

**code:** Contains the code used for this project.
* **code/luigi_code:** Contains the code for execution by luigi
* **code/luigi_status:** Contains luigi status files

**reports:** Contains all reports, graphs, and slides for this project.
* **reports/img:** Contains images (.jpg, .png) used for the DataStory jupyter notebook
* **reports/intermediary reports:** Contains additional reports regarding project

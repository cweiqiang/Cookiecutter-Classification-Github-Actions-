A cookiecutter template for the Titanic ML project
==============================

Project Title and Description
------------
This project is a machine learning project that uses the Titanic dataset to predict the survival of passengers on the Titanic. It was developed as an attempt to illustrate the usage of the cookiecutter template with the Titanic dataset and machine learning challenge, since to the best of the author's knowledge, 

The goal of the project is to build and evaluate machine learning models that can accurately predict the survival of passengers on the Titanic based on the available features in the dataset, such as age, sex, class, etc.

The project consists of several bash and Python scripts that perform the following tasks:

- Make interim datasets from the raw train and test datasets
- Build features to create processed train and test datasets from the interim datasets
- Perform data visualization and exploratory data analysis
- Train machine learning models on the processed train dataset
- Perform processing and predictions on test data
- Create feature importances visualizations for final analysis

Prerequisites
------------
Python 3
Required Python libraries: Please view requirements.txt

Getting Started
------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Two bash scripts (`run_with_gpu.sh`& `run_without_gpu.sh`) have been written for an end-to-end pipeline to make/process data, build features, perform EDA, train machine learning models (with or without GPUs), perform inference on test data, and finally create feature importances visualizations for final analysis. 

(I) You may either run the bash scripts to run all python scripts 

$ bash run_without_gpu. [non-GPU version]
$ bash run_with_gpu. [GPU version]

or 

(II) run specific Python scripts as follows:

1. Make interim dataset from raw train and test datasets:

>> python3 ./src/data/make_dataset.py --test=0 './data/raw/train.csv' './data/interim/train_interim.csv'
>> python3 ./src/data/make_dataset.py --test=1 './data/raw/test.csv' './data/interim/test_interim.csv'

2. Build features to create processed train and test datasets from interim datasets:
>> python3 ./src/features/build_features.py --test=0 './data/interim/train_interim.csv' './data/processed/train_processed.dil'
>> python3 ./src/features/build_features.py --test=1 './data/interim/test_interim.csv' './data/processed/test_processed.dil'

3. Perform in-depth EDA and visualization:

>> python3 ./src/visualization/visualize.py './data/processed/train_processed_pre-onehot.csv' './data/processed/train_processed_final.csv' './reports/figures/'

4a. Train model (without GPU):

>> python3 ./src/models/train_model_without_gpu.py './data/processed/train_processed.dil' './data/raw/train.csv'
>> rmdir -r catboost_info

4b. Train model (with GPUs):

>> python3 ./src/models/train_model_with_gpu.py './data/processed/train_processed.dil' './data/raw/train.csv'
>> rmdir -r catboost_info

5. Perform predictions on test data

6. Create Feature Importances visulizations for final analysis:

python3 ./src/visualization/feature_importances.py './data/processed/train_processed.dil' './models/' './reports/figures/'

Docker
------------
Before building and running, pls change Dockerfile contents to that of Dockerfile_without_gpu

$ docker build -t titanic_image_name .
$ docker run -it titanic_image_name:latest

To use GPUs, we shall need to follow this installation guide closely (need Nvidia-docker installation) on host machine to use its GPUs on running downloaded docker image/container https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
We also need to change Dockerfile contents to that of Dockerfile_with_gpu

$ docker build -t titanic_gpu_image_name .
$ docker run -it --gpus all titanic_gpuimage_name:latest

You may also docker pull the image from here in Dockerhub:
$

Flask API in Docker for batch inference, hosted at AWS Beanstalk
----------------------------------------------------------------------------


Streamlit webapp in Docker, hosted at AWS Beanstalk
---------------------------------------------------------------------------------

Authors
------------
Wayne Chen

License
------------
This project is licensed under the MIT License - see the LICENSE file for details.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

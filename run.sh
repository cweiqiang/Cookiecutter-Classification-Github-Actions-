#!/bin/bash
# A bash script that summarizes the key steps and python scripts to run, feel free to comment out commands to implement specific steps and commands
# Step 1: Make interim dataset from raw train and test datasets
python3 ./src/data/make_dataset.py --test=0 './data/raw/train.csv' './data/interim/train_interim.csv'
python3 ./src/data/make_dataset.py --test=1 './data/raw/test.csv' './data/interim/test_interim.csv'
# Step 2: Build features to create processed train and test datasets from interim datasets
python3 ./src/features/build_features.py --test=0 './data/interim/train_interim.csv' './data/processed/train_processed.dil'
python3 ./src/features/build_features.py --test=1 './data/interim/test_interim.csv' './data/processed/test_processed.dil'
# Step 3: Perform in-depth EDA and visualization
python3 ./src/visualization/visualize.py './data/processed/train_processed_pre-onehot.csv' './data/processed/train_processed_final.csv' './reports/figures/'
# Step 4: Train machine learning models on processed data
python3 ./src/models/train_model.py './data/processed/train_processed.dil' './data/raw/train.csv'
rmdir -r catboost_info
# Step 5: Perform predictions (batch-inference) on test data
# Step 6: Create Feature Importances visulizations for final analysis
python3 ./src/visualization/feature_importances.py './data/processed/train_processed.dil' './models/' './reports/figures/'


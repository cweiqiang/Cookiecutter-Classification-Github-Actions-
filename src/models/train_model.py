# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python ./src/models/train_model.py [OPTIONS] INPUT_FILEPATH1 INPUT_FILEPATH2

Assume current file location as illustrated as follows in the following:
EXAMPLE
-------
>> .
~/cookiecutter/cookiecutter-data-science-template-[Titanic]/Classification_Titanic
>> ls
data     Makefile   README.md   requirements.txt  test_environment.py
docs     models     references  setup.py          tox.ini
LICENSE  notebooks  reports     src

Uses input train_processed dataset (X) in compressed format (dil) with the dill package
and labels from original dataset (y0 to train several machine learning models to be saved in pickle,
and outputs graphs and csv file on algorithm performances on train & validation datasets:

>> python ./src/models/train_model.py './data/processed/train_processed.dil' './data/raw/train.csv'

See more of usage of Python package click at https://click.palletsprojects.com/en/8.1.x/
"""

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from sklearn.pipeline import Pipeline
import dill
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score, matthews_corrcoef

# @click.option('--test', default=0, required=True, help='0: For Train dataset, \n 1: For Test dataset.', type=int)
# @click.argument('output_filepath', type=click.Path())
@click.command()
@click.argument('input_filepath1', type=click.Path(exists=True))
@click.argument('input_filepath2', type=click.Path(exists=True))
def main(input_filepath1, input_filepath2):
    """Runs model training scripts to turn processed data from (saved in ../data/processed) into
       several models ready to be analyzed and saved in the folder (./models/).
        
       Parameters:
       input_filepath1 (str): the path of the processed dataset 
       input_filepath2 (str): the path of the training labels
       
       The script performs the following tasks:
        1. Reads the processed datasets for features and training labels
        2. Trains several models using the processed data
        3. Saves the trained models in the './models/' folder
    """
    logger = logging.getLogger(__name__)
    logger.info('Training machine learning models from processed training/validation data sets...')


    # Create the parameter grids for each classification algorithm to be used
    xgb_param_grid = {
        'clf__learning_rate': np.arange(0.05, 1.00, 0.05),
        'clf__max_depth': np.arange(3, 10, 1),
        'clf__n_estimators': np.arange(50, 200, 50)
        }

    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear', 'saga']
        }

    dt_param_grid = {
        'clf__max_depth': [2, 3, 5, 10],
        'clf__min_samples_leaf': [5, 10, 20, 50],
        'clf__criterion': ["gini", "entropy"]
        }

    knn_param_grid = {
        'clf__n_neighbors': list(range(1, 31))
        }

    lgb_param_grid = {
        'clf__learning_rate': [0.01, 0.1, 1],
        'clf__n_estimators': [100, 500, 1000],
        'clf__max_depth': [3, 5, 7],
        'clf__num_leaves': [31, 63, 127],
        'clf__lambda_l1': [0, 0.1, 1],
        'clf__lambda_l2': [0, 0.1, 1]
    }

    ctb_param_grid = {
        'clf__depth': [4, 6, 8],
        'clf__learning_rate': [0.01, 0.02, 0.04],
        'clf__iterations': [10, 20, 30]
                    }

    rf_param_grid = { 
        'clf__n_estimators': [200, 400],
        'clf__max_features': ['auto', 'sqrt', 'log2'],
        'clf__max_depth' : [4,6,8],
        'clf__criterion' :['gini', 'entropy']
    }

    pipelines = []
    # pipelines.append(('XGB', Pipeline([('clf', XGBClassifier(max_depth=3, tree_method="gpu_hist", enable_categorical=True))]), xgb_param_grid))
    pipelines.append(('XGB', Pipeline([('clf', XGBClassifier(max_depth=3, gpu_id=-1, tree_method="hist"))]), xgb_param_grid))
    pipelines.append(('LR', Pipeline([('clf', LogisticRegression(max_iter=2000, tol=0.001))]), lr_param_grid))
    pipelines.append(('DT', Pipeline([('clf', DecisionTreeClassifier())]), dt_param_grid))
    pipelines.append(('LGBM', Pipeline([('clf', LGBMClassifier())]), lgb_param_grid))
    pipelines.append(('CatB', Pipeline([('clf', CatBoostClassifier())]), ctb_param_grid))
    pipelines.append(('KNN', Pipeline([('clf', KNeighborsClassifier())]), knn_param_grid))
    pipelines.append(('RF', Pipeline([('clf', RandomForestClassifier())]), rf_param_grid))

    cv_folds = 5
    n_iter = 10
    scoring = "roc_auc"
    test_prop = 0.2

    model_names = []
    results = []
    records = []
    model_filenames = []

    # dillfile = open('./data/processed/train_processed.dil', 'rb')
    dillfile = open(input_filepath1, 'rb')
    X_pipeline = dill.load(dillfile)
    X_pipeline_array = X_pipeline.toarray()
    print(X_pipeline_array.shape)

    df = pd.read_csv(input_filepath2)
    # Drop any duplicate rows
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
    target_var = 'Survived'
    labelencoder = LabelEncoder()
    df[target_var] = labelencoder.fit_transform(df[target_var])
    df[target_var] = df[target_var].astype('category')
    y = df[target_var].to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_pipeline, y, test_size=test_prop, random_state=42)
    for pipename , pipeline, param_grid in pipelines:
        rs_cv = RandomizedSearchCV(estimator=pipeline, cv=cv_folds, n_iter=n_iter, scoring=scoring, verbose=1, param_distributions=param_grid)
        rs_cv.fit(X_train, y_train)
        crosscv_results = rs_cv.cv_results_['mean_test_score']
        model_filename = './models/' + pipename + '_' + scoring + '_rs_cv_model.sav'
        # save model with pickle
        pickle.dump(rs_cv, open(model_filename, 'wb'))
        filename = './models/' + pipename + '_' + scoring + "_rs_cv_model.joblib"
        # save model with joblib
        joblib.dump(rs_cv, filename)
        model_filenames.append(model_filename)
        results.append(crosscv_results)
        model_names.append(pipename)
        msg = "%s: Score Mean %f , Score Std (%f)" % (model_names, crosscv_results.mean(), crosscv_results.std())
        print(msg)
        records.append((model_names, crosscv_results.mean(), crosscv_results.std()))

    # Compare different Algorithms
    fig = plt.figure(figsize=[6.4*1.25, 4.8*1.25])
    fig.suptitle('Algorithm Classification Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(model_names)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel(scoring+"_score", fontsize=12)
    plt.savefig('./reports/figures/Algorithm_Training_Data_Classification_Performance_Comparison.png')
    plt.show()


    model_preds = [pickle.load(open(model_file,'rb')).predict(X_test) for model_file in model_filenames]
    models_dict = dict(zip(model_names, model_preds))
    # Create a dictionary to store the performance metrics for each model
    metrics = {}
    # Calculate the performance metrics for each model
    for name, y_pred in models_dict.items():
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        metrics[name] = {
            'Accuracy': report['accuracy'],
            'AUC': roc_auc_score(np.array(y_test), y_pred),
            'Recall': report['macro avg']['recall'],
            'Prec.': report['macro avg']['precision'],
            'F1': report['macro avg']['f1-score'],
            'Kappa': cohen_kappa_score(np.array(y_test), y_pred),
            'MCC': matthews_corrcoef(np.array(y_test), y_pred),
        }

    # Create a DataFrame from the metrics dictionary
    clf_report_df = np.round(pd.DataFrame.from_dict(metrics, orient='index'),3)

    train_results_means = np.array([records[i][1] for i in range(0,len(model_names))])
    train_results_std = np.array([records[i][2] for i in range(0,len(model_names))])
    scr = scoring.upper().split('_')[1]

    # Combing train and test prediction overall results as a dataframe, train_test_results_df
    train_results_df = np.round(pd.DataFrame({scr+' Mean':train_results_means, scr+' Std':train_results_std}, index=model_names),3)
    train_results_df.columns = pd.MultiIndex.from_product([['Train_Results'], train_results_df.columns])
    clf_report_df.columns = pd.MultiIndex.from_product([['Validation_Results'], clf_report_df.columns])
    train_test_results_df = pd.concat([train_results_df, clf_report_df], axis = 1)

    # Save the train_test_results_df dataframe train-test-prediction results to a CSV file
    train_test_results_df.to_csv('./reports/train_validation_classification_comparison_report.csv', index=True)
    return None

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python ./src/models/predict_model.py [OPTIONS] processed_test_data_filepath model_filepath report_output_filepath

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

>> python ./src/models/predict_model.py './data/processed/test_processed.dil' './models/XGB_roc_auc_rs_cv_model.sav' './reports/test_predictions_complete_xgb.csv'

See more of usage of Python package click at https://click.palletsprojects.com/en/8.1.x/
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import dill
import pickle

@click.command()
@click.argument('processed_test_data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('report_report_output_filepath', type=click.Path())
def main(processed_test_data_filepath, model_filepath, report_output_filepath):
  """Make predictions on a processed test dataset using a trained model and save the predictions.

  Parameters:
  	processed_test_data_filepath (str): The path of the processed test dataset
  	model_filepath (str): The path of the trained model file
        report_output_filepath (str): The path to save the output predictions

  The script performs the following tasks:
        1. Reads the processed test dataset from processed_test_data_filepath
        2. Loads the trained model from model_filepath
        3. Runs predictions on the test dataset using the loaded model
        4. Saves the predictions in a csv file at the report_output_filepath
  """
  logger = logging.getLogger(__name__)
  logger.info('making final data set from raw data')

  dillfile = open(processed_test_data_filepath, 'rb')
  X_test = dill.load(dillfile)
  dillfile.close()
  
  df_test = pd.read_csv('./data/raw/test.csv')
  model = pickle.load(open(model_filepath,'rb'))
  predictions = model.predict(X_test)

  test_predictions_df_complete = pd.DataFrame({'Survived': predictions})
  test_predictions_df_complete = pd.concat([df_test, test_predictions_df_complete], axis=1)
  test_predictions_df_complete.to_csv(report_output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

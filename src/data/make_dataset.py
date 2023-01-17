# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python make_dataset.py [OPTIONS] INPUT_FILEPATH OUTPUT_FILEPATH

Assume current file location as illustrated as follows in the following:
EXAMPLE
-------
>> .
~/cookiecutter/cookiecutter-data-science-template-[Titanic]/Classification_Titanic
>> ls
data     Makefile   README.md   requirements.txt  test_environment.py
docs     models     references  setup.py          tox.ini
LICENSE  notebooks  reports     src

For preprocessing input train dataset and output interim train dataset:
>> python ./src/data/make_dataset.py --test=0 './data/raw/train.csv' './data/interim/train_interim.csv'

For preprocessing input test dataset and output interim test dataset:
Ensure that the columns of train & test dataset are same/and in same order after dropping target variable
>> python ./src/data/make_dataset.py --test=1 './data/raw/test.csv' './data/interim/test_interim.csv'

See more of usage of Python package click at https://click.palletsprojects.com/en/8.1.x/
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
import pickle

@click.command()
@click.option('--test', default=0, required=True, help='0: For Train dataset, \n 1: For Test dataset.', type=int)
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(test, input_filepath, output_filepath):
    """Preprocess a raw dataset and save it as an interim dataset for analysis.

    Parameters:
        test (int): 0 for the train dataset, 1 for the test dataset
        input_filepath (str): the path of the raw dataset (a CSV file)
        output_filepath (str): the path where the interim dataset will be saved (a CSV file)

    The script performs the following tasks:
        1. Reads the input CSV file, removes duplicate rows and certain columns based on whether the input file is the train or test dataset. 
        2. Processes the resultant dataframe for missing values, converting data types, and reordering ordinal variables.
        3. Imputes missing values in the numerical and categorical columns separately using the 'median' and 'most_frequent' strategies, respectively,
        4. Saves the processed dataset in the output filepath
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Reading input CSV file
    df = pd.read_csv(input_filepath)
    
    # Dropping any duplicate rows
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
    
    # Setting the 'PassengerId' column as ID variable
    id_var = 'PassengerId'
    
    # Checking if the input file is train or test dataset
    if test == 0:
        # If train dataset, then setting the 'Survived' column as target variable
        target_var = 'Survived'
        # Dropping 'PassengerId' and 'Survived' column from the dataframe
        df = df.drop([id_var, target_var], axis=1)
    elif test == 1:
        # If test dataset, then dropping 'PassengerId' column from the dataframe
        df = df.drop([id_var], axis=1)
    else:
        pass
    
    global DataPreprocessingClass
    # Stage 1a. Data PreProcessing Stage
    class DataPreprocessingClass(BaseEstimator, TransformerMixin):
        """A class for preprocessing data for machine learning models.

        The class allows for handling missing values, converting data types, and
        reordering ordinal variables.

        Parameters
        ----------
        col : str, optional (default='Cabin')
            The name of the column to replace missing values with a symbol.
        col_replace_nan : bool, optional (default=True)
            Whether to replace missing values in the specified column with a symbol.
        col_replace_sym : str, optional (default='~')
            The symbol to use for replacing missing values in the specified column.
        ord_col : str, optional (default='Pclass')
            The name of a column to reorder as an ordinal variable.
        ordinal_columns : list of str, optional (default=['Pclass', 'SibSp', 'Parch'])
            The names of the columns to treat as ordinal variables with default ordering.
        missing_props_threshold : float, optional (default=0.6)
            The maximum proportion of missing values allowed for a column to be kept.
        df_replace_sym : str, optional (default='?')
            The symbol to replace with missing values globally in the input dataframe.

        Attributes
        ----------
        categorical_columns_ : list of str
            The names of the columns treated as categorical variables after fitting.
        float_columns_ : list of str
            The names of the columns treated as float variables after fitting.
        high_missing_cols_ : list of str
            The names of the columns with more than the specified proportion of missing values, which are dropped after fitting.

        Methods
        -------
        fit(X, y=None)
            Fits the transformer on the input data.
        transform(X, y=None)
            Transforms the input data.
        
        Examples
        --------
        >>> X = pd.DataFrame({'Age': [25, 30, 35, '?'], 'Cabin': ['A4', '?', '?', np.nan]})
        >>> preprocessor = DataPreprocessingClass()
        >>> preprocessor.fit_transform(X)
            Age Cabin
        0 25.0   A4
        1 30.0    ~
        2 35.0    ~
        3  NaN    ~
        """
        def __init__(self):
            self.col = "Cabin"
            self.col_replace_nan = True
            self.col_replace_sym = '~'
            self.ord_col = "Pclass"
            self.ordinal_columns = ["Pclass", "SibSp", "Parch"]
            self.missing_props_threshold = 0.6
            self.df_replace_sym = '?'
            pass

        def fit(self, X, y=None):
            """Fit the preprocessor to the data.

            Parameters
            ----------
            X : pandas DataFrame
                The data to fit the preprocessor to.
            y : pandas Series, optional (default=None)
                The target values. This parameter is not used in this class.

            Returns
            -------
            self : DataPreprocessingClass
                The fitted preprocessor.

            Notes
            -----
            The preprocessor modifies the following attributes after fitting:
            categorical_columns_, float_columns_, high_missing_cols_.
            """
            X_c = X.copy()
            if self.col_replace_nan == True:
                X_c[self.col].replace(np.nan, self.col_replace_sym, inplace=True)
            else:
                pass
            # Create a boolean mask for categorical columns
            self.categorical_feature_mask = ((X_c.dtypes == object)|(X_c.dtypes == 'category')|(X_c.dtypes == 'bool'))
            # Store list of categorical column names from X
            self.categorical_columns = X_c.columns[self.categorical_feature_mask].tolist()

            # Create a boolean mask for float columns
            self.float_feature_mask = (X_c.dtypes == float)
            # Store list of float column names from X
            self.float_columns = X_c.columns[self.float_feature_mask].tolist()

            # Drop columns that have higher than 60% missing values  
            missing_props = X_c.isnull().mean()
            high_missing = missing_props[missing_props > self.missing_props_threshold]
            self.high_missing_cols = list(high_missing.index)
            return self

        def transform(self, X):
            """Transform the data with the fitted preprocessor.

            Parameters
            ----------
            X : pandas DataFrame
                The data to transform.

            Returns
            -------
            X_t : pandas DataFrame
                The transformed data.

            Notes
            -----
            The transformer assumes that the input dataframe has the same structure as
            the one used to fit the preprocessor.
            """
            X_t = X.copy()
            # Replace selected symbol globally within dataframe with np.nan
            X_t.replace(self.df_replace_sym, np.nan, inplace=True)
            if self.col in list(X_t.columns):
                X_t[self.col].replace(np.nan, self.col_replace_sym, inplace=True)
            else:
                pass
            # Ensure all supposed categorical variables are set as categorical with no ordering.
            X_t[self.categorical_columns] = X_t[self.categorical_columns].astype(CategoricalDtype(ordered=False))
            # Set ordinal variables with default ordering in classes
            X_t[self.ordinal_columns] = X_t[self.ordinal_columns].astype(CategoricalDtype(ordered=True))
            # Set special ordinal variables with reordering in classes, e.g. Pclass 3 < 2 < 1
            X_t[self.ord_col] = X_t[self.ord_col].astype(CategoricalDtype([3, 2, 1], ordered=True))
            X_t.drop(self.high_missing_cols, axis=1, inplace=True)
            return X_t

    global DataframeFeatureImputerUnion
    class DataframeFeatureImputerUnion(BaseEstimator, TransformerMixin):
        """A class for imputing missing values in a dataframe and combining numerical and categorical features.

        The class imputes missing values in the numerical and categorical columns
        separately using the 'median' and 'most_frequent' strategies, respectively,
        and then combines the imputed dataframes.

        Parameters
        ----------
        ord_col : str
            The name of the ordinal variable with special ordering.
        ordinal_columns : list of str
            The names of the ordinal variables with default ordering.

        Attributes
        ----------
        categorical_columns_ : list of str
            The names of the categorical columns in the input data.
        float_columns_ : list of str
            The names of the float columns in the input data.
        cat_mapper_ : DataFrameMapper
            The DataFrameMapper object used to impute missing values in the categorical columns.
        float_mapper_ : DataFrameMapper
            The DataFrameMapper object used to impute missing values in the float columns.
        
        Methods
        -------
        fit(X, y=None)
            Fits the transformer on the input data.
        transform(X, y=None)
            Transforms the input data.

        Examples
        --------
        >>> X = pd.DataFrame({'age': [25, 30, 35, 40], 'income': [50000, 55000, 60000, np.nan], 'gender': ['M', 'F', 'M', 'F']})
        >>> imputer = DataframeFeatureImputerUnion(ord_col='age', ordinal_columns=['income'])
        >>> X_imputed = imputer.fit_transform(X)
        >>> X_imputed
        age  income gender
        0 25.0  50000.0      M
        1 30.0  55000.0      F
        2 35.0  60000.0      M
        3 40.0  60000.0      F
        """
        def __init__(self):
            self.ord_col = "Pclass"
            self.ordinal_columns = ["Pclass", "SibSp", "Parch"]
            pass

        def fit(self, X, y=None):
            """Fit the imputer to the data.

            Parameters
            ----------
            X : pandas DataFrame
                The data to fit the imputer to.
            y : pandas Series, optional (default=None)
                The target values. This parameter is not used in this class.

            Returns
            -------
            self : DataframeFeatureImputerUnion
                The fitted imputer.

            Notes
            -----
            The imputer modifies the following attributes after fitting:
            categorical_columns_, float_columns_, cat_mapper_, float_mapper_.
            """
            # Create a boolean mask for categorical columns
            self.categorical_feature_mask = (X.dtypes == 'category')
            # Store list of categorical column names from X
            self.categorical_columns = X.columns[self.categorical_feature_mask].tolist()
            # Create a boolean mask for float columns
            self.float_feature_mask = (X.dtypes == float)
            # Store list of float column names from X
            self.float_columns = X.columns[self.float_feature_mask].tolist()

            # Apply float imputer
            float_imputation_mapper = DataFrameMapper([([float_feature], SimpleImputer(strategy="median")) for float_feature in self.float_columns],
                                                        input_df=True,
                                                        df_out=True
                                                    )
            # Apply categorical imputer
            categorical_imputation_mapper = DataFrameMapper([([category_feature], SimpleImputer(strategy="most_frequent")) for category_feature in self.categorical_columns],
                                                            input_df=True,
                                                            df_out=True)                                            
            float_imputation_mapper.fit(X)
            categorical_imputation_mapper.fit(X)

            self.float_mapper = float_imputation_mapper
            self.cat_mapper = categorical_imputation_mapper
            return self

        def transform(self, X, y=None):
            """Impute missing values in the numerical and categorical columns and combine the imputed dataframes.

            Parameters
            ----------
            X : pandas DataFrame
                The data to impute and combine.

            Returns
            -------
            X_imputed : pandas DataFrame
                The data with imputed values and combined numerical and categorical columns.
            
            """
            float_df = self.float_mapper.transform(X)
            cat_df = self.cat_mapper.transform(X)
            X_imputed = pd.concat([float_df, cat_df], axis=1)

            X_imputed[self.float_columns] = X_imputed[self.float_columns].astype('float')
            # Ensure all supposed categorical variables are set as categorical with no ordering.
            X_imputed[self.categorical_columns] = X_imputed[self.categorical_columns].astype(CategoricalDtype(ordered=False))
            # Set ordinal variables with default ordering in classes
            X_imputed[self.ordinal_columns] = X_imputed[self.ordinal_columns].astype(CategoricalDtype(ordered=True))
            # Set special ordinal variables with reordering in classes, e.g. Pclass 3 < 2 < 1
            X_imputed[self.ord_col] = X_imputed[self.ord_col].astype(CategoricalDtype([3, 2, 1], ordered=True))
            return X_imputed

    global data_processing_pipeline1
    data_processing_pipeline1 = Pipeline(steps=[
        ("Preprocess_Data", DataPreprocessingClass()),
        ("Feature_Union_Imputer", DataframeFeatureImputerUnion())
        ])

    if test == 0:
        X_pipeline1_df = data_processing_pipeline1.fit_transform(df)
        X_pipeline1_df.to_csv(output_filepath, index=False)
        # create a pickle file
        picklefile = open('./data/raw/train_data_processing_pipeline1.pkl', 'wb')
        # pickle the dictionary and write it to file
        pickle.dump(data_processing_pipeline1, picklefile, pickle.HIGHEST_PROTOCOL)
        # close the file
        picklefile.close()
    elif test == 1:
        # read the pickle file
        picklefile = open('./data/raw/train_data_processing_pipeline1.pkl', 'rb')
        # unpickle the dataframe
        data_processing_pipeline = pickle.load(picklefile)
        # close file
        picklefile.close()
        X_pipeline1_df = data_processing_pipeline.transform(df)
        X_pipeline1_df.to_csv(output_filepath, index=False)
    else: 
        print('Your file is not preprocessed properly, pls key in 0, 1 for --test option')
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

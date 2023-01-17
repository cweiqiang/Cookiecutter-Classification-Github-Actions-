# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python ./src/features/build_features.py [OPTIONS] INPUT_FILEPATH OUTPUT_FILEPATH 

Assume current file location as illustrated as follows in the following:
EXAMPLE
-------
>> .
~/cookiecutter/cookiecutter-data-science-template-[Titanic]/Classification_Titanic
>> ls
data     Makefile   README.md   requirements.txt  test_environment.py
docs     models     references  setup.py          tox.ini
LICENSE  notebooks  reports     src

For feature engineering input train_interim dataset (csv) 
and output train_processed dataset in compressed format (dil) with the dill package:

>> python ./src/features/build_features.py --test=0 './data/interim/train_interim.csv' './data/processed/train_processed.dil'

For feature engineering input test_interim dataset (csv) 
and output test_processed dataset in compressed format (dil) with the dill package:
Ensure that the columns of train & test dataset are same/and in same order after dropping target variable

>> python ./src/features/build_features.py --test=1 './data/interim/test_interim.csv' './data/processed/test_processed.dil'

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
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer
import dill
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

@click.command()
@click.option('--test', default=0, required=True, help='0: For Train dataset, \n 1: For Test dataset.', type=int)
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(test, input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (./data/interim) into
       cleaned data ready to be analyzed (saved in ../data/processed).
       
       Parameters:
       	test (int): 0 for the train dataset, 1 for the test dataset
       	input_filepath (str): the path of the interim dataset (a CSV file)
       	output_filepath (str): the path where the processed dataset will be saved (a CSV file)
       
       The script performs the following tasks:
        1. Reads the input CSV file
        2. Processes text data using the TextProcessingClass()
        3. Performs Continuous Feature Engineering using ContinuousFeatureEngineering()
        4. Performs Categorical Feature Engineering using CategoricalFeatureEngineering()
        5. Finalizes Feature Variables using FinalizeFeatureVars()
        6. Scales float variables using DataframeFeatureScalerUnion()
        7. Apply Dictifier() to convert the dataframe to a dictionary
        8. Apply DictVectorizer(sort=False) to perform one-hot encoding on the dictionary and obtain a compressed sparse row (csr) matrix for model training.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath)

    global TextProcessingClass
    class TextProcessingClass(BaseEstimator, TransformerMixin):
        """A class for processing text data for machine learning models.
        The class allows for extracting substrings from text data, vectorizing the
        extracted substrings, and adding the vectorized data as additional features
        to the input dataframe.
        
        Parameters
        ----------
        col : str, optional (default='Cabin')
        The name of the column to extract substrings from.
        
        module : module, optional (default=re)
        The Python module to use for extracting substrings.
        
        re_method : callable, optional (default=re.findall)
        The method from the specified module to use for extracting substrings.
        
        pattern : str, optional (default='[A-Z]')
        The pattern to use for extracting substrings.
        
        Attributes
        ----------
        feature_names_ : list of str
        The names of the extracted substrings after fitting.
        vocab_dict_ : dict
        A dictionary mapping the extracted substrings to integer indices after fitting.
        
        Methods
        -------
        fit(X, y=None)
        Fits the transformer on the input data.
        
        transform(X, y=None)
        Transforms the input data.
        
        Examples
        --------
        >>> X = pd.DataFrame({'Cabin': ['A4', 'B5', 'C6', 'D7']})
        >>> processor = TextProcessingClass()
        >>> processor.fit_transform
           Cabin Cabin=A Cabin=B Cabin=C
           0   A4     1     0     0     0
           1   B4     0     1     0     0
           2   C4     0     0     1     0
           3  None    0     0     0     0
        """
        def __init__(self):
            self.col = "Cabin"
            self.module = __import__('re')
            self.re_method = getattr(self.module, 'findall')
            self.pattern = r"[A-Z]"
        pass
    
        def fit(self, X, y=None):
            """Fit the transformer to the training data.
            This method is used to fit the transformer to the training data, which
            involves extracting any needed parameters from the input data.
            Parameters
            ----------
            X : pandas DataFrame
            The input data for fitting the transformer.
            
            y : None
            This parameter is not used in this transformer.
            
            Returns
            -------
            self : TextProcessingClass
            The transformer, with any learned parameters from the input data.
            """
            X_c = X.copy()
            X_c['col_temp'] = X_c[self.col].apply(lambda x: ''.join([match for match in self.re_method(self.pattern, x)]))
            vectorizer_c = CountVectorizer(lowercase=False, ngram_range=(1, 1), analyzer='char')
            bow_matrix = vectorizer_c.fit_transform(X_c['col_temp'])
            # Store within self, `vocab_dict` &  `feature_names` from vectorized train set, 
            # to be used to transform both train & test sets later
            self.vocab_dict = vectorizer_c.vocabulary_
            self.feature_names = vectorizer_c.get_feature_names_out()
            return self

        def transform(self, X):
            """Transform the input data.
            This method is used to transform the input data using the fitted transformer.
            
            Parameters
            ----------
            X : pandas DataFrame
            The input data for transforming.
            
            Returns
            -------
            X_transformed : pandas DataFrame
            The transformed input data.
            """
            X_t = X.copy()
            X_t['col_temp'] = X_t[self.col].apply(lambda x: ''.join([match for match in self.re_method(self.pattern, x)]))
            vectorizer_t = CountVectorizer(lowercase=False, ngram_range=(1, 1), analyzer='char', vocabulary=self.vocab_dict)
            bow_matrix = vectorizer_t.transform(X_t['col_temp'])
            # Generate matrix of word vectors
            bow_array = bow_matrix.toarray()
            bow_X = pd.DataFrame(bow_array)
            bow_X = bow_X.astype('category')
            # bow_X.columns = self.feature_names
            bow_X.columns = [self.col + '=' + feat_name for feat_name in self.feature_names]
            X_transformed = pd.concat([X_t, bow_X], axis=1)
            X_transformed.drop('col_temp', axis=1, inplace=True)
            return X_transformed
    
    global ContinuousFeatureEngineering
    class ContinuousFeatureEngineering(BaseEstimator, TransformerMixin):
        """A class for engineering features from continuous variables.
        The class generates quantile groups for each float variable, and adds the
        resulting grouping columns to the input data.
        
        Parameters
        ----------
        kmax : int, optional (default=10)
            The maximum number of quantile groups to create for each float variable.
            
        Attributes
        ----------
        float_columns_ : list of str
        The names of the float variables in the input data.

        cols_quantile_edges_ : list of array-like
        The quantile edges used to create the quantile groups for each float variable.

        cols_group_names_ : list of list of str
        The names of the quantile groups for each float variable.
        
        Methods
        -------
        find_elbow(self, wcss):
        Find the "elbow" in the WCSS curve.
        
        fit(X, y=None)
        Fits the transformer on the input data.
        
        transform(X, y=None)
        Transforms the input data.
        
        Examples
        --------
        >>> X = pd.DataFrame({'Age': [25, 30, 35, 40, 45], 'Income': [50000, 60000, 70000, 80000, 90000]})
        >>> cts_feat_engineer = ContinuousFeatureEngineering()
        >>> cts_feat_engineer.fit_transform(X)
        Age   Income Age_group Income_group
        0 25.0  50000.0    25-30       50000.0-70000.0
        1 30.0  60000.0    25-30       50000.0-70000.0
        2 35.0  70000.0    30-40       70000.0-80000.0
        3 40.0  80000.0    40-50       80000.0-90000.0
        """
        def __init__(self):
            self.kmax = 10
            pass
        
        def find_elbow(self, wcss):
            """Find the "elbow" in the WCSS curve, which is the point of inflection where adding more clusters no longer significantly decreases WCSS.
            Parameters
            ----------
            wcss : list or numpy array
            The within-cluster sum of squares (WCSS) values for different numbers of clusters.
            
            Returns
            -------
            cluster_num : int
            A best estimate of the optimal number of clusters corresponding to the elbow in the WCSS curve.
            """
            # Calculate the 1st order finite differences between consecutive WCSS values
            diff = np.diff(wcss) 
            # padded with two '1.0' to account for loss of 1st two indices, this means we are aiming for k>=3
            ratio_diff = np.concatenate((np.array([1.0, 1.0]), diff[1:]/diff[:-1])) 
            # Return the number of clusters corresponding to the elbow index
            cluster_num = np.argmin(ratio_diff) + 1
            return cluster_num
        
        def fit(self, X, y=None):
            """Fit the transformer.
            
            Parameters
            ----------
            X : pandas DataFrame
            The input dataframe to fit.
            
            y : None
            There is no need for a target in a transformer, yet the pipeline API requires this parameter.
            
            Returns
            -------
            self : object
            Returns self.
            """
            X_c = X.copy()
            float_feature_mask = (X_c.dtypes == 'float')
            # Get list of float variable column names (string-like variables that have)
            self.float_columns = X_c.columns[float_feature_mask].tolist()
            self.cols_quantile_edges = []
            self.cols_group_names = []
            
            for col in self.float_columns:
                grp_col = col + '_group'
                data = X_c[col].fillna(X_c[col].median()).values
                data_reshaped = data.reshape(-1,1)
                # Fit k-means clustering models with different values of k
                wcss = [KMeans(n_clusters=k, max_iter=500, random_state=200).fit(data_reshaped).inertia_ for k in range(1, self.kmax+1)]
                wcss_opt_k = self.find_elbow(wcss)
                # Generate the quantile edges using pandas.qcut with the number of quantiles (i.e. the number of bins)
                quantile_edges = np.floor(pd.qcut(data, wcss_opt_k, retbins=True, duplicates='drop')[1])
                quantile_edges[-1] = np.ceil(X_c[col].max())
                # Use the quantile edges to generate the quantile labels
                group_names = ['{}-{}'.format(int(quantile_edges[i]), int(quantile_edges[i+1]) if i < len(quantile_edges)-2 else int(quantile_edges[i+1])) for i in range(len(quantile_edges)-1)]
                X_c[grp_col] = pd.cut(X_c[col], quantile_edges, labels=group_names, include_lowest=True)
                self.cols_quantile_edges.append(quantile_edges)
                self.cols_group_names.append(group_names)
                # Store the (float_columns, cols_quantile_edges, cols_group_names) to be reloaded for transforming test-sets
            return self

        def transform(self, X):
            """Transform the input dataframe.
            
            Parameters
            ----------
            X : pandas DataFrame
            The input dataframe with float `col` variable values to perform binning transformation.
            
            Returns
            -------
            X_t : pandas DataFrame
            The transformed dataframe with new column `col`+'_group' containing the binned values after applying binning 
            on float `col` variable values with quantile edges and group names fitted on training data.
            """
            X_t = X.copy()
            n_float_cols = len(self.cols_group_names)
            for i in range(0, n_float_cols):
                col = self.float_columns[i]
                quantile_edges = self.cols_quantile_edges[i]
                group_names = self.cols_group_names[i]
                grp_column = col + '_group'
                X_t[grp_column] = pd.cut(X_t[col], quantile_edges, labels=group_names, include_lowest=True)

            return X_t
    
    global CategoricalFeatureEngineering    
    class CategoricalFeatureEngineering(BaseEstimator, TransformerMixin):
        """A class for engineering new categorical features from existing ones.
        The class allows for extracting the title from the 'Name' column and adding
        a new column indicating whether a value is missing in the 'Ticket' column.
            
        Parameters
        ----------
        cat_col : str, optional (default='Name')
        The name of the column to extract the title from.
            
        new_cat_col : str, optional (default='Title')
        The name of the column to store the extracted title in.
        col : str, optional (default='Ticket')
            
        The name of the column to check for missing values.
        isnull_col : str, optional (default='Has_Ticket_Info')
            
        The name of the column to store a boolean indicating whether a value is
        missing in the specified column.
            
        Attributes
        ----------
        None
            
        Methods
        -------
        fit(X, y=None)
        Fits the transformer on the input data.
            
        get_title(self, name):
        Extract and return the title from the name.
            
        replace_titles(self, X):
        Replace certain titles with more general titles.
            
        transform(X, y=None)
        Transforms the input data.
            
        Examples
        --------
        >>> X = pd.DataFrame({'Name': ['Mr. John Smith', 'Mrs. Jane Smith', 'Miss. Mary Johnson'],
        ...                   'Ticket': ['123456', np.nan, '234567']})
        >>> feature_engineer = CategoricalFeatureEngineering()
        >>> feature_engineer.fit_transform(X)
        Name                 Ticket      Title   Has_Ticket_Info
        0 Mr. John Smith        123456      Mr      True
        1 Mrs. Jane Smith       NaN         Mrs     False
        2 Miss. Mary Johnson    234567      Miss    True
        """
        def __init__(self):
            self.cat_col = 'Name'
            self.new_cat_col = 'Title'
            self.col = 'Ticket'
            self.isnull_col = 'Has_Ticket_Info'
            pass
            
        def fit(self, X, y=None):
            """Fit the transformer on the training data.
            Parameters
            ----------
            X : pd.DataFrame, shape (n_samples, n_features)
            The training data.
            y : pd.Series, shape (n_samples,), optional
            The target values.
                
            Returns
            -------
            self : CategoricalFeatureEngineering
            The transformer object.
            """
            return self
                
        def get_title(self, name):
            """Extract and return the title from the name.
            Parameters
            ----------
            name : str
            The name from which to extract the title.
                
            Returns
            -------
            title : str
            The extracted title.
            """
            if '.' in name:
                return name.split(',')[1].split('.')[0].strip()
            else:
                return 'Unknown'
            
        def replace_titles(self, X):
            """Replace certain titles with more general titles.
                
            Parameters
            ----------
            X : pd.DataFrame, shape (n_samples, n_features)
            The dataframe containing the title column.
                
            Returns
            -------
            title : str
            The modified title.
            """
            title = X[self.new_cat_col]
            if title in ['Capt', 'Col', 'Major']:
                return 'Officer'
            elif title in ["Jonkheer", "Don", 'the Countess', 'Dona', 'Lady',"Sir"]:
                return 'Royalty'
            elif title in ['the Countess', 'Mme', 'Lady']:
                return 'Mrs'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            else:
                return title
                
        def transform(self, X):
            """Transform input data by extracting and engineering new categorical features.
            
            Parameters
            ----------
            X : pd.DataFrame
            The input dataframe to be transformed.
            
            Returns
            -------
            X_t: pd.DataFrame
            The transformed dataframe with new categorical features.
            """
            X_t = X.copy()
            X_t[self.new_cat_col] = X_t[self.cat_col].map(lambda row: self.get_title(row))
            X_t[self.new_cat_col] = X_t.apply(self.replace_titles, axis=1)
            X_t[self.new_cat_col] = X_t[self.new_cat_col].astype('category')
            X_t[self.isnull_col] = X_t[self.col].isnull()
            X_t[self.isnull_col] = X_t[self.isnull_col].astype('category')
            return X_t
    
    global FinalizeFeatureVars
    class FinalizeFeatureVars(BaseEstimator, TransformerMixin):
        """FinalizeFeatureVars is a transformer class that finalizes the feature variables for model training. 
        It removes insignificant columns specified and retains a dataframe of categorical and float columns for later use.
        
        Parameters
        ----------
        None
        
        Attributes
        ----------
        insign_cols : list
        List of insignificant columns to be removed from the data.
        
        categorical_columns : list
        List of categorical column names from the data.
        
        float_columns : list
        List of float column names from the data.
        
        Methods
        -------
        fit(self, X, y=None)
        Fits the transformer to the data and stores the list of insignificant columns.
        
        transform(self, X)
        Removes insignificant columns and stores lists of categorical and float column names.
        """
        def __init__(self):
            pass
        
        def fit(self, X, y=None):
            """
            Fits the transformer to the data and stores the list of insignificant columns.
            
            Parameters
            ----------
            X : pandas DataFrame
            Data to fit the transformer to.
            
            y : pandas Series, optional
            Target variable. Not used in this transformer.
            
            Returns
            -------
            self : FinalizeFeatureVars
            Returns the instance of the transformer.
            """
            self.insign_cols = ["Name", "Ticket", "Cabin"]
            return self
            
        
        def transform(self, X):
            """Transform input dataframe X by dropping insignificant columns, storing lists of categorical and float column names.
            
            Parameters
            ----------
            X : pandas dataframe, shape (n_samples, n_features)
            Input dataframe.
            
            Returns
            -------
            X_t : pandas dataframe, shape (n_samples, n_features - len(insignificant_columns))
            Transformed dataframe with insignificant columns dropped.
            """
            X_t = X.copy()
            X_t.drop(self.insign_cols, axis=1, inplace=True)
            # Create a boolean mask for categorical/ordinal columns
            categorical_feature_mask = ((X_t.dtypes == 'category'))
            # Store list of categorical column names from X
            self.categorical_columns = X_t.columns[categorical_feature_mask].tolist()
            
            # Create a boolean mask for float columns
            float_feature_mask = (X_t.dtypes == float)
            # Store list of float column names from X
            self.float_columns = X_t.columns[float_feature_mask].tolist()
            return X_t
    
    global DataframeFeatureScalerUnion
    class DataframeFeatureScalerUnion(BaseEstimator, TransformerMixin):
        """Custom transformer class for scaling and combining float and categorical features in a dataframe.
        
        Parameters
        ----------
        ordinal_columns : list
        List of ordinal column names.
        
        special_ord_col : str
        Name of special ordinal column that requires reordering.
        
        Attributes
        ----------
        categorical_columns : list
        List of categorical column names from input data.
        
        categorical_feature_mask : boolean mask
        Boolean mask for identifying categorical columns in input data.
        
        float_columns : list
        List of float column names from input data.
        
        float_feature_mask : boolean mask
        Boolean mask for identifying float columns in input data.
        
        float_scaling_mapper : DataFrameMapper
        DataFrameMapper object for scaling float features.
        
        categorical_identity_mapper : DataFrameMapper
        DataFrameMapper object for applying identity transformation on categorical features.
        
        Methods
        -------
        fit(X, y=None)
        Fits the transformer on the input data.
        
        transform(X, y=None)
        Transforms the input data.
        
        Examples
        --------
        >>> X = pd.DataFrame({'Age': np.array([25.0, 30.0, 35.0]), 'Cabin': np.array(['A1', 'B2', 'C3']), 'Pclass': np.array([ 1, 2, 3]),'SibSp':np.array([0, 2, 1]), 'Parch': np.array([1, 2, 0]) })
        >>> df_feat_scaler_union = DataframeFeatureScalerUnion()
        >>> print(df_feat_scaler_union.fit_transform(input_dataframe))
        Age Cabin Pclass SibSp Parch
        0 -1.224745    A1      1     0     1
        1  0.000000    B2      2     2     2
        2  1.224745    C3      3     1     0
        """
        def __init__(self):
            self.ordinal_columns = ['Pclass', 'SibSp', 'Parch']
            self.special_ord_col = "Pclass"
            pass
        
        def fit(self, X, y=None):
            """Fit the transformer on the input data.
            
            Parameters
            ----------
            X : pandas dataframe
            Input data.
            
            y : None
            Dummy parameter to adhere to scikit-learn transformer convention.
            
            Returns
            -------
            self : DataframeFeatureScalerUnion
            Returns an instance of self.
            """
            # Create a boolean mask for categorical columns
            self.categorical_feature_mask = ((X.dtypes == 'category')|(X.dtypes == 'object')|(X.dtypes == 'bool')|(X.dtypes == 'int'))
            # Store list of categorical column names from X
            self.categorical_columns = X.columns[self.categorical_feature_mask].tolist()
            # Create a boolean mask for float columns
            self.float_feature_mask = (X.dtypes == float)
            # Store list of float column names from X
            self.float_columns = X.columns[self.float_feature_mask].tolist()
            
            # Apply standard scaling mapper on float features
            float_scaling_mapper = DataFrameMapper([([float_feature], StandardScaler()) for float_feature in self.float_columns],
                                                    input_df=True,
                                                    df_out=True
                                                   )
            # Apply identity mapper on categorical features
            categorical_identity_mapper = DataFrameMapper([([category_feature], FunctionTransformer()) for category_feature in self.categorical_columns],
                                                        input_df=True,
                                                        df_out=True)
            float_scaling_mapper.fit(X)
            categorical_identity_mapper.fit(X)
            self.float_scaling_mapper = float_scaling_mapper
            self.categorical_identity_mapper = categorical_identity_mapper
            return self
            
        def transform(self, X, y=None): 
            """
            Scale float features and combine unchanged categorical features of dataframe X into a dataframe X_scaled.
            
            Parameters
            ----------
            X : pandas dataframe
            Input dataframe.
            
            y : None
            Dummy parameter to adhere to scikit-learn transformer convention.
            
            Returns
            -------
            X_scaled: pandas dataframe
            Transformed dataframe with float features scaled and unchanged categorical/ordinal features combined.
            """
            float_df = self.float_scaling_mapper.transform(X)
            cat_df = self.categorical_identity_mapper.transform(X)
            X_scaled = pd.concat([float_df, cat_df], axis=1)
            # This assumes categorical_mapper is stored at last entry of mappers
            # # Ensure all supposed categorical variables are set as categorical with no ordering.
            X_scaled[self.float_columns] = X_scaled[self.float_columns].astype('float')
            # Ensure all supposed categorical variables are set as categorical with no ordering.
            X_scaled[self.categorical_columns] = X_scaled[self.categorical_columns].astype(CategoricalDtype(ordered=False))
            # Set ordinal variables with default ordering in classes
            X_scaled[self.ordinal_columns] = X_scaled[self.ordinal_columns].astype(CategoricalDtype(ordered=True))
            # Set special ordinal variables with reordering in classes, e.g. Pclass 3 < 2 < 1
            X_scaled[self.special_ord_col] = X_scaled[self.special_ord_col].astype(CategoricalDtype([3, 2, 1], ordered=True))
            return X_scaled

    global Dictifier
    class Dictifier(BaseEstimator, TransformerMixin):
        """Converts a DataFrame or list of dictionaries to a list of dictionaries.
        
        Parameters
        ----------
        None
        
        Attributes
        ----------
        None
        
        Methods
        -------
        fit(self, X, y=None)
        Fits the transformer to the data.
        
        Parameters
        ----------
        X : pd.DataFrame or list of dictionaries
            The input data to fit.
            
        y : any, optional
            Ignored.
            
        Returns
        -------
        self : Dictifier object
            The fitted transformer.
        
        transform(self, X)
        Transforms the input data to a list of dictionaries.
        
        Parameters
        ----------
        X : pd.DataFrame or list of dictionaries
            The input data to transform.
            
        Returns
        -------
        list of dictionaries which each dictionary corresponding to a row of the input dataframe X
        with its keys, values containing the columns, and colume variable values respectively.
        
        Examples
        -------
        >>> X = pd.DataFrame({'Age': np.array([20, 30]), 'Pclass': np.array([3,1])})
        >>> dictifier_class = Dictifier()
        >>> print(dictifier_class.fit_transform(X))
        [{'Age': 20, 'Pclass': 3}, {'Age': 30, 'Pclass': 1}]
        """     
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            if type(X) == pd.core.frame.DataFrame:
                return X.to_dict("records")
            else:
                return pd.DataFrame(X).to_dict("records")
    
    global DictVectorizer
    global data_processing_pipeline1
    global data_processing_pipeline2
    global X_pipeline
    global X_pipeline2
    global X_feature_names

    data_processing_pipeline1 = Pipeline(
        steps=[("Process_Text", TextProcessingClass()),
        ("Cts_Var_FeatureEng", ContinuousFeatureEngineering()),
        ("Cat_Var_Feature_Eng", CategoricalFeatureEngineering()),
        ("Finalize_Feature_Vars", FinalizeFeatureVars()),
        ("Scale_Float_Vars", DataframeFeatureScalerUnion())]
        )

    data_processing_pipeline2 = Pipeline(
        steps=[("Process_Text", TextProcessingClass()),
        ("Cts_Var_FeatureEng", ContinuousFeatureEngineering()),
        ("Cat_Var_Feature_Eng", CategoricalFeatureEngineering()),
        ("Finalize_Feature_Vars", FinalizeFeatureVars()),
        ("Scale_Float_Vars", DataframeFeatureScalerUnion()),
        ("Dictifier", Dictifier()),
        ("Dict_Vectorizer", DictVectorizer(sort=False))
        ]
        )
    
    
    if test == 0:
        # X_pipeline1: pandas dataframe produced by data_processing_pipeline1 
        # (before Dicitifier & DictVectorizer to perform one-hot encoding to produce csr matrix).
        X_pipeline1_df = data_processing_pipeline1.fit_transform(df)
        X_pipeline1_df.to_csv('./data/processed/train_processed_pre-onehot.csv', index=False)
        # create a dill file to store the `data_processing_pipeline1` object fitted with train data df
        dillfile0 = open('./data/interim/train_data_processing_pipeline1.dil', 'wb')
        # dill the pipeline1 object and write it to file
        dill.dump(data_processing_pipeline1, dillfile0)
        dillfile0.close()
        
        # X_pipeline2: compressed sparse row (csr) format matrix of type `<class 'numpy.float64>`.
        X_pipeline2 = data_processing_pipeline2.fit_transform(df)
        # create a dill file to store the `data_processing_pipeline2` object fitted with train data df
        dillfile = open('./data/interim/train_data_processing_pipeline2.dil', 'wb')
        # dill the pipeline2 object and write it to file
        dill.dump(data_processing_pipeline2, dillfile)
        # close the file
        dillfile.close()

        dillfile2 = open(output_filepath, 'wb')
        # dill the csr numpy matrix from fit-transformed train data and write it to file
        dill.dump(X_pipeline2, dillfile2)
        # close the file
        dillfile2.close()

        # X_feature_names : feature names of the output dataframe from step `Dictifier()` fitted with train data
        # to be used later in feature importance analysis during prediction phase
        X_feature_names = list(data_processing_pipeline2[-1].get_feature_names_out())
        dillfile3 = open('./data/interim/X_feature_names.dil', 'wb')
        # dill the feature_names from transformed train data and write it to file
        dill.dump(X_feature_names, dillfile3)
        dillfile3.close()
        X_pipeline2_df = pd.DataFrame(X_pipeline2.toarray(), columns=X_feature_names)
        X_pipeline2_df.to_csv('./data/processed/train_processed_final.csv', index=False)

    elif test == 1:
        # read & undill the dill file to load the `data_processing_pipeline1` object fitted with train data
        dillfile0 = open('./data/interim/train_data_processing_pipeline1.dil', 'rb')  
        data_processing_pipeline1 = dill.load(dillfile0)
        # close file
        dillfile0.close()
        X_pipeline1_df = data_processing_pipeline1.transform(df)
        X_pipeline1_df.to_csv('./data/processed/test_processed_pre-onehot.csv', index=False)

        
        dillfile = open('./data/interim/train_data_processing_pipeline2.dil', 'rb')  
        data_processing_pipeline2 = dill.load(dillfile)
        # close file
        dillfile.close()

        # Use loaded `data_processing_pipeline2` object fitted with train data to transform test-data to
        # ensure same schema in output in X_pipeline: compressed sparse row (csr) format matrix of type `<class 'numpy.float64>`.
        X_pipeline2 = data_processing_pipeline2.transform(df)
        dillfile2 = open(output_filepath, 'wb')
        # dill the csr numpy matrix from transformed test data and write it to file
        dill.dump(X_pipeline2, dillfile2)
        # close the file
        dillfile2.close()

        X_feature_names = list(data_processing_pipeline2[-1].get_feature_names_out())
        X_pipeline2_df = pd.DataFrame(X_pipeline2.toarray(), columns=X_feature_names)
        X_pipeline2_df.to_csv('./data/processed/test_processed_final.csv', index=False)

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

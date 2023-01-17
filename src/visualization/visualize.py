# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python ./src/visualization/visualize.py [OPTIONS] pre_oneshot_input_filepath final_processed_input_filepath report_output_dirpath

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

>> python ./src/visualization/visualize.py './data/processed/train_processed_pre-onehot.csv' './data/processed/train_processed_final.csv' './reports/figures/'

See more of usage of Python package click at https://click.palletsprojects.com/en/8.1.x/
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import dill
import pickle
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import phik 
from phik import resources, report


@click.command()
@click.argument('pre_oneshot_input_filepath', type=click.Path(exists=True))
@click.argument('final_processed_input_filepath', type=click.Path(exists=False))
@click.argument('report_output_dirpath', type=click.Path(dir_okay=True, exists=True))
def main(pre_oneshot_input_filepath, final_processed_input_filepath, report_output_dirpath):
    """
    Create visualizations of two processed train datasets and save them in the report_output_dirpath.

    Parameters:
        pre_oneshot_input_filepath (str): The path of the first processed train dataset
        final_processed_input_filepath (str): The path of the second processed train dataset
        report_output_dirpath (str): The path of the directory where the visualization files will be saved

    The script performs the following tasks:
        1. Reads the first processed train dataset from pre_oneshot_input_filepath
        2. Reads the second processed train dataset from final_processed_input_filepath
        3. Creates various visualizations comparing the two datasets
        4. Saves the visualizations in the report_output_dirpath/reports subfolder
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating visualizations from processed datasets...')
    df_pre1hot = pd.read_csv(pre_oneshot_input_filepath)
    df_final = pd.read_csv(final_processed_input_filepath)
    df_raw = pd.read_csv('./data/raw/train.csv')
    # Drop any duplicate rows
    df_raw.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)

    target_var = 'Survived'
    id_var = 'PassengerId'
    labelencoder = LabelEncoder()
    df_raw[target_var] = labelencoder.fit_transform(df_raw[target_var])
    df_raw[target_var] = df_raw[target_var].astype('category')
    y = df_raw[target_var].to_numpy().ravel()
    
    df_pre1hot['Survived'] = y
    df_final['Survived'] = y


    print(df_pre1hot.sample(5))
    print(df_pre1hot.info())

    print(df_final.sample(5))
    print(df_final.info())

    # Create a boolean mask for categorical columns
    cts_feature_mask = (df_pre1hot.dtypes == float)
    # Store list of categorical column names from X
    cts_variables = df_pre1hot.columns[cts_feature_mask].tolist()
    cat_ord_vars = df_pre1hot.columns[~cts_feature_mask].tolist()

    #### Continuous Variable Distributional Analysis with respect to target variable
    # Calculate the number of rows and columns needed for the subplots
    import numpy as np
    n = len(cts_variables)
    cols = 5
    rows = int(np.ceil(n / cols))

    # Create a figure with the subplots
    fig, axs = plt.subplots(rows, cols, figsize=(20, 8), sharey=True)
    if rows == 1:
        axs = axs.reshape(1, cols)

    # Iterate over the dataframes and create a crosstab plot for each one
    for i in np.arange(0,n):
        # Get the index of the subplot
        row = i // cols # row : quotient
        col = i % cols  # col : remainder

        # Create the distributional plot in the subplot
        sns.kdeplot(data=df_pre1hot, x=cts_variables[i], hue=target_var, ax=axs[row, col], multiple='stack');

    for j in np.arange(n%cols,cols):
        axs[n//cols,j].set_visible(False)

    # Add a title to the figure
    fig.suptitle('Distributional plots of Continuous Variables against Target Variable')
    fig_filepath = report_output_dirpath + 'Distributional_plots_of_Continuous_Variables_against_Target_Variable.png'
    plt.savefig(fig_filepath)



    #### Categorical Variable Crosstab Analysis with respect to target variable
    # Crosstabs plots for categorical variables with respect to target variable
    plt.figure(figsize=(20, 32));
    # Calculate the number of rows and columns needed for the subplots
    n = len(cat_ord_vars)
    rows = int(np.ceil(n / 5))
    cols = 5

    # Create a figure with the subplots
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 8), sharey=True)
    if rows == 1:
        axs = axs.reshape(1,cols)

    # Iterate over the dataframes and create a crosstab plot for each one

    for i in np.arange(0, n):
        # Get the index of the subplot
        row = i // cols # row : quotient
        col = i % cols  # col : remainder

        # Create the crosstab plot in the subplot
        pd.crosstab(df_pre1hot[cat_ord_vars[i]], df_pre1hot[target_var], normalize=True).plot(kind='bar', ax=axs[row, col], rot=0);

    for j in np.arange(n%cols,cols):
        axs[n//cols,j].set_visible(False)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4);
    # Add a title to the figure
    fig.suptitle('Crosstab plots for categorical/ordinal variables against target variable');
    fig_filepath = report_output_dirpath + 'Crosstab_plots_for_categorical_ordinal_variables_against_target_variable.png'
    plt.savefig(fig_filepath)


    ## Pairwise Variable Analysis to determine groups of correlated variables with Phi-k 
    # (a generalised correlation measure [range 0-1] for pairs of interval, categorical/ordinal variables)
    df_phik_matrix = df_final.phik_matrix(interval_cols=cts_variables)
    matrix_filepath = report_output_dirpath + 'Phi-k_correlation_matrix.csv'
    df_phik_matrix.to_csv(matrix_filepath, index=True)
    
    df_phik_targetvar = np.round(df_phik_matrix[['Survived']].sort_values(by='Survived', ascending=False, axis=0).iloc[1:20,:],3)
    array_filepath = report_output_dirpath + 'Phik-k-Top-20-correlated-features-with-target-var.csv'
    df_phik_targetvar.to_csv(array_filepath, index=True)

    phik.report.plot_correlation_matrix(df_phik_matrix.values, x_labels=list(df_final.columns), y_labels=list(df_final.columns), print_both_numbers=False, figsize=(16,8), vmin=0, vmax=1, color_map='YlOrRd', title='Phi-k Correlation Matrix of Feature Variables and Target Variable')
    fig_filepath = report_output_dirpath + 'Phi-k_correlation_matrix_plot.png'
    plt.savefig(fig_filepath)
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html


    top_feat_vars = ['Title=Mr', 'Fare', 'Age']
    for feat_var in top_feat_vars:
        df_phik_feat_var = np.round(df_phik_matrix[[feat_var]].sort_values(by=feat_var, ascending=False, axis=0).iloc[1:21,:],3)
        array_filepath = report_output_dirpath + 'Phik-k-Top-20-correlated-features-with-feat-var-' + feat_var + '.csv'
        df_phik_feat_var.to_csv(array_filepath, index=True)

    # Correlation for pairs of continuous variables with Pearson's correlation coefficients
    ## As baseline reference, we still compute corr matrix with Pearson (despite some most variables being ordinal/categorical)
    ## It also yields the sign of correlation unlike phi-k
    plt.figure(figsize=(32, 16))
    mask = np.triu(np.ones_like(df_final.corr(), dtype=bool))
    heatmap=sns.heatmap(df_final.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG');
    heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
    fig_filepath = report_output_dirpath + 'Pearson_correlation_matrix_plot.png'
    plt.savefig(fig_filepath)
    # plt.show()

    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(df_final.corr()[['Survived']].sort_values(by='Survived', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Pearson-Top-20-correlated-features-with-target-var', fontdict={'fontsize':18}, pad=16);
    fig_filepath = report_output_dirpath + 'Pearson-Top-20-correlated-features-with-target-var.png'
    plt.savefig(fig_filepath, dpi=300, bbox_inches='tight')

    return print('Visualizations creation complete, please check ./reports/figures/folder')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

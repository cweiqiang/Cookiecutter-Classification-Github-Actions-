# -*- coding: utf-8 -*-
"""Usage in command line: 
>> python ./src/visualization/feature_importances.py [OPTIONS] pre_oneshot_input_filepath final_processed_input_filepath report_output_dirpath

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

>> python ./src/visualization/feature_importances.py './data/processed/train_processed.dil' './models/' './reports/figures/'

See more of usage of Python package click at https://click.palletsprojects.com/en/8.1.x/
"""
import dill
import matplotlib.pyplot as plt
import shap
import logging

# constants
TREE_MODELS_LIST = ['XGB','LGBM','CatB', 'DT', 'RF']

#function to save figures
def save_fig(fig, report_output_dirpath, model_name, fig_name):
    report_output_filepath = f"{report_output_dirpath}/{model_name}_{fig_name}.png"
    fig.savefig(report_output_filepath)
    plt.clf()

def main(X_pipeline, final_data_processing_pipeline, models_dirpath, report_output_dirpath):
    """
    Create feature importance visualizations for a processed train dataset and a set of models.

    Parameters:
        X_pipeline (str): The processed train dataset in csr format
        final_data_processing_pipeline (str): The data processing pipeline 
        models_dirpath (str): The path of the directory containing the models
        report_output_dirpath (str): The path of the directory where the visualization files will be saved

    The script performs the following tasks:
        1. Reads the processed train dataset in csr format from input_filepath1
        2. Loads the models from models_dirpath
        3. Creates feature importance visualizations based on each model itself and its corresponding SHAP version
        4. Saves the visualizations in the report_output_dirpath/reports subfolder     
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating feature importances visualizations from trained models and processed datasets...')

    X_feature_names = list(final_data_processing_pipeline[-1].get_feature_names_out())
    X_df = pd.DataFrame(X_pipeline.toarray(), columns=X_feature_names)

    ## Tree-based Models (XGB, LGBM, DT, RF, CatB) #########
    for tree_model in TREE_MODELS_LIST:
        model_filepath = models_dirpath + tree_model + '_roc_auc_rs_cv_model.sav'
        auc_model = pickle.load(open(model_filepath,'rb'))
        sorted_idx1 = auc_model.best_estimator_.named_steps["clf"].feature_importances_.argsort()
        plt.barh(final_data_processing_pipeline[-1].get_feature_names_out()[sorted_idx1], auc_model.best_estimator_.named_steps["clf"].feature_importances_[sorted_idx1])
        plt.xlabel(tree_model + " model Feature Importances")
        save_fig(plt, report_output_dirpath, tree_model, "Model_Feature_Importances")

        auc_explainer = shap.TreeExplainer(auc_model.best_estimator_.named_steps["clf"])
        auc_shap_values = auc_explainer.shap_values(X_pipeline.toarray())
        shap.summary_plot(auc_shap_values, X_pipeline.toarray(), feature_names=X_feature_names, max_display=10, show=False)
        save_fig(plt, report_output_dirpath, tree_model, "Model_SHAP_Feature_Importances2")
        
        
        auc_explainer = shap.TreeExplainer(auc_model.best_estimator_.named_steps["clf"])
        auc_shap_values = auc_explainer.shap_values(X_pipeline.toarray())
        #xgb_auc_explainer = shap.TreeExplainer(xgb_auc_model.best_estimator_.named_steps["clf"])
        #xgb_auc_shap_values = xgb_auc_explainer.shap_values(X_pipeline)
        shap.summary_plot(auc_shap_values, X_pipeline.toarray(), feature_names=X_feature_names, max_display=10, show=False)
        # shap.summary_plot(xgb_auc_shap_values, X_pipeline.toarray(), feature_names=X_feature_names, max_display=10, show=False)
        report_output_filepath2 = report_output_dirpath + tree_model +'_Model_SHAP_Feature_Importances2.pdf'
        # report_output_filepath2 = report_output_dirpath + tree_model +'_SHAP_Feature_Importances2.pdf'
        plt.savefig(report_output_filepath2)
        # plt.savefig(report_output_filepath2)
        plt.close()
        # plt.close()
        # 
        # shap.summary_plot(xgb_auc_shap_values, X_df, plot_type="bar", max_display=10, show=False)
        shap.summary_plot(auc_shap_values, X_df, plot_type="bar", max_display=10, show=False)
        report_output_filepath = report_output_dirpath + tree_model + '_Model_SHAP_Feature_Importances.pdf'
        plt.savefig(report_output_filepath)
        plt.close()
        
        

    ## Linear Models #########


    model_filepath = models_dirpath + 'LR_roc_auc_rs_cv_model.sav'
    lr_auc_model = pickle.load(open(model_filepath,'rb'))
    # Get the feature coefficients
    coefficients = lr_auc_model.best_estimator_.steps[-1][1].coef_[0]
    # Sort the feature coefficients in descending order
    sorted_coefficients = sorted(zip(X_feature_names, coefficients), key=lambda x: x[1], reverse=False)
    # Get the top-ranked features
    top_features = sorted_coefficients
    # Extract the feature names and coefficients
    feature_names, coefficients = zip(*top_features)
    # Set the figure size
    plt.figure(figsize=(8, 6))
    # Draw a horizontal bar chart
    plt.barh(feature_names, coefficients)
    # Set the x-axis label
    plt.xlabel('Coefficient')
    report_output_filepath = report_output_dirpath + 'LR_Feature_Importances.png'
    plt.savefig(report_output_filepath)
    plt.close()

    masker =  shap.maskers.Independent(data=X_pipeline)
    lr_auc_explainer = shap.LinearExplainer(lr_auc_model.best_estimator_.named_steps["clf"], masker=masker)
    lr_auc_shap_values = lr_auc_explainer.shap_values(X_pipeline)
    X_df = pd.DataFrame(X_pipeline.toarray(), columns=X_feature_names)
    shap.summary_plot(lr_auc_shap_values, X_df, plot_type="bar", max_display=10, show=False);
    report_output_filepath = report_output_dirpath + 'LR_SHAP_Feature_Importances.pdf'
    plt.savefig(report_output_filepath)
    plt.close()




    return print('Feature Importances visualizations created, please check [./reports/figures/] folder')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()







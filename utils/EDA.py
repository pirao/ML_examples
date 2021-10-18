import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import ppscore as pps

from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, mean_absolute_error, r2_score

from yellowbrick.regressor import residuals_plot, prediction_error
from yellowbrick.model_selection import learning_curve, feature_importances, FeatureImportances

import xgboost as xgb
import eli5
from eli5.sklearn import PermutationImportance

################################
#   Univariate correlation
################################

def corr_plot(df,target_col,th=0.4):
    """Visualization of all 3 univariate correlations (Pearson, Spearman,PPS)

    Args:
        df (pandas): Pandas dataframe containing the dataset
        target_col (str): Target column of the dataset
        th (float, optional): Display a threshold to better highlight good and bad correlations. Defaults to 0.4.
    """
    c_vec = []
    methods = ['pearson', 'spearman', 'pps']

    # Correlation methods between the dataset and the target variable
    for m in methods:
        if m != 'pps':
            corr = df.corr(method=m).drop(columns=target_col).loc[target_col, :].values
            c_vec.append(corr)
        else:
            corr_data = pps.matrix(df)[['x', 'y', 'ppscore']].pivot( columns='x', index='y', values='ppscore')
            corr = corr_data.loc[target_col, :].drop(columns=target_col).values
            c_vec.append(corr)

    cols = df.drop(columns=target_col).columns

    plot_df = pd.DataFrame()
    for i in range(len(c_vec)):
        aux_df = pd.DataFrame()
        aux_df['cols'] = cols
        aux_df['corr'] = c_vec[i]
        aux_df['method'] = methods[i]

        if plot_df.shape[0] != 0:
            plot_df = pd.concat([plot_df, aux_df])
        else:
            plot_df = aux_df

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=plot_df, x='cols', y='corr', hue='method', ax=ax)
    
    if th:
        x = [ax.get_xticks()[0]-1, ax.get_xticks()[-1]+1]
        ax.plot(x, [th, th], 'r', label=r'$\rho = $|{}|'.format(th))
        ax.plot(x, [-th, -th], 'r')
        ax.set_xlim(x)

    ax.set_xlabel('Variables')
    ax.set_ylabel('Correlation coefficient')
    ax.tick_params(labelsize=20, rotation=45)



def dual_heat_map(data, figsize=(25, 15), dual=True):
    """Plot the Pearson and Spearman heatmap side by side

    Args:
        data (pandas): Pandas dataframe containing the data
        figsize (tuple, optional): Figure size. Defaults to (25, 15).
        dual (bool, optional): Specify whether the visualize just the Pearson heatmap or both Pearson and Spearman side by side. Defaults to True.
    """
    sns.set(font_scale=1.1)
    corr_pearson = data.corr(method='pearson')
    corr_spearman = data.corr(method='spearman')

    mask = np.zeros_like(corr_pearson)
    mask[np.triu_indices_from(mask)] = True

    if dual:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        sns.heatmap(corr_pearson, cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size": 14}, mask=mask, square=True, ax=ax[0], fmt='.2f', cbar=False)
        sns.heatmap(corr_spearman, cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size": 14}, mask=mask, square=True, ax=ax[1], fmt='.2f', cbar=False)
        ax[0].set_title('Pearson correlation')
        ax[1].set_title('Spearman correlation')
        plt.show()

    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.heatmap(corr_pearson, cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={
                    "size": 14}, mask=mask, square=True, fmt='.2f', cbar=False)
        ax.set_title('Pearson correlation')
        plt.show()

    return


def pps_heat_map(data, figsize=(30, 15)):
    """Power predictive score heatmap

    Args:
        data (pandas): Pandas dataframe
        figsize (tuple, optional): Figure size. Defaults to (30, 15).
    """
    corr = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, annot=True)
    plt.title('Power predictive score (PPS)')
    plt.show()
    return

################################################
#       Pairplots
##############################################


def scatterplot_pearson(df, x_vars, y_vars, cmap='viridis', hue='Class', height=3, aspect=1):
    """Pairplot of all variables of a dataframe

    Args:
        df (pandas): Pandas dataframe
        x_vars (list of variable names): Variables to plot on the x axis
        y_vars (list of variable names): Variables to plot on the y axis
        cmap (str, optional): Colormap of the plot. Defaults to 'viridis'.
        hue (str, optional): Variable in df to map plot aspects to different colors. Defaults to 'Class'.
        height (int, optional): Height (in inches) of each facet.. Defaults to 3.
        aspect (int, optional): Aspect * height gives the width (in inches) of each facet.. Defaults to 1.

    Returns:
        Pairgrid: Returns the underlying PairGrid instance for further tweaking.
    """
    sns.set(font_scale=1.9)
    g = sns.PairGrid(df, hue=hue, x_vars=x_vars, y_vars=y_vars,palette=cmap, corner=False, height=height, aspect=aspect)
    g.map_diag(sns.histplot, color='.5')
    g.map_offdiag(sns.scatterplot, s=3, alpha=0.6)
    g.add_legend(fontsize=16, bbox_to_anchor=(1.5, 1))
    g.tight_layout()
    plt.setp(g._legend.get_title(), fontsize='20')  # for legend title
    return g


##############################################
# Multivariate correlation
###############################################


class multivariate_importance():
    def __init__(self, X_train, X_test, y_train, y_test, nmodels=6):
        """Object used for multivariate correlation based around wrapper methods of specific ML models. 

        Args:
            X_train (pandas): Training set containing input variables
            X_test (pandas): Test set containing input variables
            y_train (pandas): Training set containing output variables
            y_test (pandas): Test set containing output variables
            nmodels (int, optional): Number of models to use and compare feature importance. Defaults to 6.
        """
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nmodels = nmodels

        # List of all models used
        mod1 = Lasso()
        mod2 = RandomForestRegressor(random_state=0, n_jobs=-1)
        mod3 = AdaBoostRegressor(random_state=0)
        mod4 = GradientBoostingRegressor(random_state=0)
        mod5 = ExtraTreesRegressor(random_state=0, n_jobs=-1)
        mod6 = xgb.XGBRegressor(seed=123, gpu_id=0, tree_method='gpu_hist', random_state=0, n_jobs=-1)

        self.mod_list = [mod1, mod2,
                         mod3, mod4,
                         mod5, mod6]

        self.mod_list = self.mod_list[0:self.nmodels]

        self.model_r2 = None

        print('All models for determining feature importance')
        print(self.mod_list)
        print('')

    def train_models(self):
        """Train all specified models

        Returns:
            list: List containing the R2 metric of each model
        """
        model_r2 = []
        for model in tqdm(self.mod_list):
            model.fit(self.X_train, self.y_train)
            model_r2.append(
                np.round(r2_score(self.y_test, model.predict(self.X_test)), 4))

        self.model_r2 = model_r2

        return model_r2

    def permutation_importance(self, model_index=1):
        """Visualize the importance of each column based on their permutation. More info in https://www.kaggle.com/learn/machine-learning-explainability

        Args:
            model_index (int, optional): Model index tp specify which model to study. Goes from 1 to 6.  Defaults to 1.

        Returns:
            pandas: dataframe ranking each column based on feature importance
        """

        self.mod_list[model_index].fit(self.X_train, self.y_train)
        perm = PermutationImportance(self.mod_list[model_index], random_state=1).fit(self.X_train, self.y_train)
        return eli5.show_weights(perm, feature_names=X_train.columns.tolist())

    def plot(self, relative=True, topn=8, absolute=True, plot_R2=True):
        """Plot the feature importance of all specified models

        Args:
            relative (bool, optional): Normalize the feature importance by the largest feature importance. Defaults to True.
            topn (int, optional): Number of top features displayed. Defaults to 8.
            absolute (bool, optional): Specify whether to display the absolute feature importance or not. Defaults to True.
            plot_R2 (bool, optional): Display the R2 metric of each model as a title. Defaults to True.
        """
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 18))

        if self.model_r2 == None:
            print('Obtaining R2 score for all 6 models')
            multivariate_importance.train_models(self)
            print('R2 score calculated')

        # Obtaining feature importance of each model
        print('Obtaining feature importance - 0%')
        viz1 = FeatureImportances(self.mod_list[0], relative=relative, topn=topn, ax=ax[0, 0], absolute=absolute)
        viz1.fit(self.X_train, self.y_train)
        ax[0, 0].tick_params(labelsize=18)

        viz2 = FeatureImportances(self.mod_list[1], relative=relative, topn=topn, ax=ax[0, 1], absolute=absolute)
        viz2.fit(self.X_train, self.y_train)
        ax[0, 1].tick_params(labelsize=18)

        viz3 = FeatureImportances(self.mod_list[2], relative=relative, topn=topn, ax=ax[0, 2], absolute=absolute)
        viz3.fit(self.X_train, self.y_train)
        ax[0, 2].tick_params(labelsize=18)
        print('Obtaining feature importance - 50%')
        viz4 = FeatureImportances(self.mod_list[3], relative=relative, topn=topn, ax=ax[1, 0], absolute=absolute)
        viz4.fit(self.X_train, self.y_train)
        ax[1, 0].tick_params(labelsize=18)

        viz5 = FeatureImportances(self.mod_list[4], relative=relative, topn=topn, ax=ax[1, 1], absolute=absolute)
        viz5.fit(self.X_train, self.y_train)
        ax[1, 1].tick_params(labelsize=18)

        viz6 = FeatureImportances(self.mod_list[5], relative=relative, topn=topn, ax=ax[1, 2], absolute=absolute)
        viz6.fit(self.X_train, self.y_train)
        ax[1, 2].tick_params(labelsize=13)
        print('Obtaining feature importance - 100%')

        # Display the R2 score as a title
        if plot_R2:

            ax[0, 0].set_title(
                'Lasso Regression - $R^2$ = {}'.format(self.model_r2[0]), fontsize=22)
            ax[0, 1].set_title(
                'RandomForestRegressor - $R^2$ = {}'.format(self.model_r2[1]), fontsize=22)
            ax[0, 2].set_title(
                'AdaBoostRegressor - $R^2$ = {}'.format(self.model_r2[2]), fontsize=22)
            ax[1, 0].set_title(
                'GradientBoostingRegressor - $R^2$ = {}'.format(self.model_r2[3]), fontsize=22)
            ax[1, 1].set_title(
                'ExtraTreesRegressor - $R^2$ = {}'.format(self.model_r2[4]), fontsize=22)
            ax[1, 2].set_title(
                'XGBoost - $R^2$ = {}'.format(self.model_r2[5]), fontsize=22)

            ax[0, 0].set_xlabel('Coefficient value', fontsize=22)
            ax[0, 1].set_xlabel('Coefficient value', fontsize=22)
            ax[0, 2].set_xlabel('Coefficient value', fontsize=22)
            ax[1, 0].set_xlabel('Coefficient value', fontsize=22)
            ax[1, 1].set_xlabel('Coefficient value', fontsize=22)
            ax[1, 2].set_xlabel('Coefficient value', fontsize=22)

        plt.tight_layout()
        return

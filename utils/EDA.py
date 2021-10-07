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
    sns.set(font_scale=1.9)
    g = sns.PairGrid(df, hue=hue, x_vars=x_vars, y_vars=y_vars,
                     palette=cmap, corner=False, height=height, aspect=aspect)
    g.map_diag(sns.histplot, color='.5')
    g.map_offdiag(sns.scatterplot, s=3, alpha=0.6)
    g.add_legend(fontsize=16, bbox_to_anchor=(1.5, 1))
    g.tight_layout()
    plt.setp(g._legend.get_title(), fontsize='20')  # for legend title
    return g


##############################################
# Multivariate importance
###############################################


class multivariate_importance():
    def __init__(self, X_train, X_test, y_train, y_test, nmodels=6):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nmodels = nmodels

        mod1 = Lasso()
        mod2 = RandomForestRegressor(random_state=0, n_jobs=-1)
        mod3 = AdaBoostRegressor(random_state=0)
        mod4 = GradientBoostingRegressor(random_state=0)
        mod5 = ExtraTreesRegressor(random_state=0, n_jobs=-1)
        mod6 = xgb.XGBRegressor(
            seed=123, gpu_id=0, tree_method='gpu_hist', random_state=0, n_jobs=-1)

        self.mod_list = [mod1, mod2,
                         mod3, mod4,
                         mod5, mod6]

        self.mod_list = self.mod_list[0:self.nmodels]

        self.model_r2 = None

        print('All models for determining feature importance')
        print(self.mod_list)
        print('')

    def train_models(self):

        model_r2 = []
        for model in tqdm(self.mod_list):
            model.fit(self.X_train, self.y_train)
            model_r2.append(
                np.round(r2_score(self.y_test, model.predict(self.X_test)), 4))

        self.model_r2 = model_r2

        return model_r2

    def permutation_importance(self, model_index=1):

        self.mod_list[model_index].fit(self.X_train, self.y_train)
        perm = PermutationImportance(self.mod_list[model_index], random_state=1).fit(
            self.X_train, self.y_train)
        return eli5.show_weights(perm, feature_names=X_train.columns.tolist())

    def plot(self, relative=True, topn=8, absolute=True, plot_R2=True):

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 18))

        if self.model_r2 == None:
            print('Obtaining R2 score for all 6 models')
            multivariate_importance.train_models(self)
            print('R2 score calculated')

        print('Obtaining feature importance - 0%')
        viz1 = FeatureImportances(
            self.mod_list[0], relative=relative, topn=topn, ax=ax[0, 0], absolute=absolute)
        viz1.fit(self.X_train, self.y_train)
        ax[0, 0].tick_params(labelsize=18)

        viz2 = FeatureImportances(
            self.mod_list[1], relative=relative, topn=topn, ax=ax[0, 1], absolute=absolute)
        viz2.fit(self.X_train, self.y_train)
        ax[0, 1].tick_params(labelsize=18)

        viz3 = FeatureImportances(
            self.mod_list[2], relative=relative, topn=topn, ax=ax[0, 2], absolute=absolute)
        viz3.fit(self.X_train, self.y_train)
        ax[0, 2].tick_params(labelsize=18)
        print('Obtaining feature importance - 50%')
        viz4 = FeatureImportances(
            self.mod_list[3], relative=relative, topn=topn, ax=ax[1, 0], absolute=absolute)
        viz4.fit(self.X_train, self.y_train)
        ax[1, 0].tick_params(labelsize=18)

        viz5 = FeatureImportances(
            self.mod_list[4], relative=relative, topn=topn, ax=ax[1, 1], absolute=absolute)
        viz5.fit(self.X_train, self.y_train)
        ax[1, 1].tick_params(labelsize=18)

        viz6 = FeatureImportances(
            self.mod_list[5], relative=relative, topn=topn, ax=ax[1, 2], absolute=absolute)
        viz6.fit(self.X_train, self.y_train)
        ax[1, 2].tick_params(labelsize=13)
        print('Obtaining feature importance - 100%')

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

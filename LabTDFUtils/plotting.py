import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

###############################
# Matplotlib
###############################

def lineplot_sensors(df, x, y, n_cols=3, figsize=(30, 40), hue=None, hue_order=None):

    n_sensors = len(y)
    n_rows = np.ceil(n_sensors/n_cols).astype('int')

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0

    for s in y:
        sns.lineplot(data=df, x=x, y=s,ax=ax[iax], hue=hue, hue_order=hue_order)
        iax += 1

    return fig, ax


def boxplot_sensors(df, x, y, n_cols=3, figsize=(30, 40), hue=None, hue_order=None):

    n_sensors = len(y)
    n_rows = np.ceil(n_sensors/n_cols).astype('int')

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0

    for s in y:
        sns.boxplot(data=df, x=x, y=s,ax=ax[iax], hue=hue, hue_order=hue_order)
        iax += 1

    return fig, ax

def kdeplot_sensors(df,  x, y, n_cols=2, figsize=(30, 40), hue=None, hue_order=None):
    
    n_sensors = len(x)
    n_rows = np.ceil(n_sensors/n_cols).astype('int')

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0

    print(ax)
    for s in x:
        sns.kdeplot(data=df, x=s, y=y,ax=ax[iax], hue=hue, hue_order=hue_order)
        iax += 1

    return fig, ax


def plot_samples(df, n_cols=2, n_rows=2, figsize=(15,8)):
    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)

    ax1.plot(df[df.columns[0]], label=df.columns[0])
    ax1.set_ylabel('Eigenvector ' + str(df.columns[0]))

    ax2.plot(df[df.columns[-2]], label=df.columns[-2])
    ax2.set_ylabel('Eigenvector ' + str(df.columns[-2]))

    ax4.plot(df[df.columns[1]], label=df.columns[1])
    ax4.set_ylabel('Autovector ' + str(df.columns[1]))
    ax4.set_xlabel('Number of timesteps')

    ax5.plot(df[df.columns[-1]], label=df.columns[-1])
    ax5.set_ylabel('Autovector ' + str(df.columns[-1]))
    ax5.set_xlabel('Number of timesteps')

    plt.tight_layout()
    plt.show()


###############################################
# Anomaly detection / Pattern changes
###############################################

def show_pattern_changes(df, x, cols, label_col, index=-1, nrows=6, ncols=1, figsize=(15, 25)):

    df_pattern = df.loc[df[label_col] == index]

    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 25), tight_layout=True)
    ax = ax.reshape((nrows*ncols,))
    iax = 0

    for s in tqdm(cols):
        sns.lineplot(data=df, x=x, y=s, ax=ax[iax], hue=None, alpha=0.7)
        sns.scatterplot(data=df_pattern, x=x, y=s,ax=ax[iax], hue=label_col, palette='Set1')
        ax[iax].legend(['All data points', 'Points identified'])
        iax += 1

    return fig, ax


####################################
#  Autocorrelation
####################################

# For random data, autocorrelations should be near zero for all lags(white noise). 
# Non-random data have at least one significant lag. When the data are not random,
# it’s a good indication that you need to use a time series analysis or incorporate lags into a regression analysis
# to model the data appropriately.

# Looking again into stationarity, our signal must:

# - Not have a trend (mean=0)
# - Have constant variance
# - Have a constant autocorrelation pattern(below the confidence interval)
# - Have no seasonal pattern.

# The autocorrelation function(ACF) declines to near zero rapidly for a stationary time series.
# For a non-stationary time series, the ACF drops slowly.
# There is also no seasonal components since there are no repetitive peaks in defined multiples for stationary time series.

# The confidence interval of 95 % is represented by the shaded cone.
# Values outside the cone suggest very likely correlation and not a statistical fluke.

# Let us now take a closer look at the partial autocorrelation(PACF).
# Instead of finding correlations of present values with lags like ACF,
# PACF finds correlation of the residuals with the next lag. It is a function that
# measures the incremental benefit of adding another lag. So if through the PACF function we
# discover that there is hidden information in the residual that can be modeled by the next lag,
# we might get a good correlation, and we will keep that next lag as a feature while modeling.

# As mentioned before, an autoregression is a model based on the assumption that present values
# of a time series can be obtained using previous values of the same time series: the present value
# is a weighted average of its past values. In order to avoid multicollinear features for time series models, 
# it is necessary to find optimum features or order of the autoregression process using the PACF plot,
# as it removes variations explained by earlier lags, so we get only the relevant features.



def plot_autocorrelation(df, i=0, zoom=False, acf_lim=100, pacf_lim=20):
    fig, ax = plt.subplots(1, 2, figsize=(25, 5))

    plot_acf(df[df.columns[i]], lags=len(df)-1, ax=ax[0])
    ax[0].set_title('Autocorrelation - Eigenvector ' + str(i))
    ax[0].set_xlabel('Number of lags')
    ax[0].set_ylabel('Pearson coefficient')

    plot_pacf(df[df.columns[i]], lags=200, method='ywmle', alpha=.5, ax=ax[1])
    ax[1].set_title('Partial autocorrelation - Eigenvector ' + str(i))
    ax[1].set_xlabel('Number of lags')
    ax[1].set_ylabel('Pearson coefficient for residual')

    if zoom == True:
        ax[0].set_xlim([0, acf_lim])
        ax[1].set_xlim([0, pacf_lim])

    return


##########################
# Visualize stationarity
##########################

def visualize_stationarity(series,window=500):
    plt.plot(series[window:], label='Series')
    plt.plot(series.rolling(window).std(), label='Std')
    plt.plot(series.rolling(window).mean(), label='Mean')
    
    plt.legend()
    return


def get_quads(df):
    quadlen = int(len(df) * 0.25)
    ss = df[:quadlen].describe()
    ss[1] = df[quadlen:quadlen*2].describe()
    ss[2] = df[quadlen*2:quadlen*3].describe()
    ss[3] = df[quadlen*3:].describe()
    return ss

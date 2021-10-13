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


####################################
#  Plotly
####################################





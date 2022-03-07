import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm




########################
# Feature extraction
########################


# With the following function we can 

def correlation(dataset, threshold):
    """Function used to select highly correlated features and remove the first feature that is correlated with any other feature
    Args:
        dataset (pandas): Pandas dataframe
        threshold (float): Threshold used to discard redundant columns. Value recomended between 0.9 and 1 but can be between 0 and 1.

    Returns:
        pandas: Filtered pandas dataframe column without redundante variables
    """
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested in absolute coeff value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

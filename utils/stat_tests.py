from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm

###############################
# Stationarity of a time series - Dickey Fuller test
###############################

# This is the statistical test that we run to determine if a time series is stationary or not.
# Without going into the technicalities of the Dickey-Fuller test, it tests the null hypothesis that a unit root is present.

# If it is, then p > 0.05, and the process is not stationary.
# Otherwise, p < 0.05, the null hypothesis is rejected, and the process is considered to be stationary.

# When the test statistic is lower than the critical value shown, 
# you reject the null hypothesis and infer that the time series is stationary.



def ADF_test(timeseries, dataDesc):
    print(' > Is {} stationary?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='t-stat')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Lags used = {:.3f}'.format(dftest[2]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is{} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))
        
    return

#for i in range(15):
#    ADF_test(df[df.columns[i]],'Autovector ' + str(i))
#    print('')


##################################
# Feature selection
##################################


##################################
# Comparing models
##################################

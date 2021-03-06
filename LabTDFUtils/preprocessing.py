import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, fft2,ifft2
from scipy.signal import welch, butter, filtfilt


from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedShuffleSplit, TimeSeriesSplit, RepeatedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler


def get_file_names(folder_path,returns=True):
    """This function obtain a list of all files inside a specific folder

    Args:
        folder_path (string): path where your desired folder is located

    Returns:
        list: list of files inside directory
    """
    
    old_path = os.getcwd()
    
    # Change directory path
    os.chdir(folder_path)
    new_path = os.getcwd()

    # Obtain files inside new directory
    file_names = os.listdir(os.path.join(new_path))
    print('File names: ', file_names)
    print('Number of files: ', len(file_names))
    
    # return to original directory
    if returns:
        os.chdir(old_path)
    
    return file_names


def import_datasets(file_names, import_file_name=False):
    """Function to import all datasets contained inside a list

    Args:
        file_names (list): List containing .csv file names
        import_file_name (bool, optional): Add file name as a column in the dataset. Defaults to False.

    Returns:
        dataframe: pandas dataframe that joins all the datasets contained in 'file_names'
    """
    df = pd.DataFrame()
    
    for csv in file_names:
        frame = pd.read_csv(csv)
        
        if import_file_name:
            frame['filename'] = os.path.basename(csv)
            
        df = df.append(frame,ignore_index=True)
        
    return df


def split_columns(df, column, new_col_names, rename_cols = True, drop_col=True, symbol='-'):
    """Function used to split a column into multiple other

    Args:
        df (dataframe): pandas dataframe
        column (string): string containing column name that you would like to split
        new_col_names (list): list containing the names of the newly created columns
        rename_cols (bool, optional): Condition to rename the newly created columns. Defaults to True.
        drop_col (bool, optional): Condition to drop the column used for extraction. Defaults to True.
        symbol (str, optional): Symbol to be used to split the data. Defaults to '-'.

    Returns:
        dataframe: dataframe with the newly created columns extracted from the strings contained inside another column
    """
    
    series = df[column]
    df_expanded = series.str.split(pat=symbol, expand=True)
    
    if rename_cols:
        df_expanded.columns=new_col_names
        
    if drop_col:
        df.drop(column,inplace=True,axis=1)
    
    return pd.concat([df,df_expanded],axis=1)
    

###################################
# Date time columns
#######################################

def cut_date_decimal_places(df, column, position=-5):

    df_aux = df.copy()
    date_strings = df_aux[column].to_numpy()

    for idx in range(len(df_aux)):
        date_strings[idx] = date_strings[idx][0:position]

        df_cut = pd.DataFrame(date_strings)

    return df_cut


########################################
# Split dataset
#####################################

def create_scaled_dataset(X, y, test_size=0.2,shuffle=True):
    """[summary]

    Args:
        X (pandas): Input dataset
        y (pandas): Target variable
        test_size (float, optional): Size of the test set. Defaults to 0.2.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.

    Returns:
        [type]: Scaled datasets split into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=shuffle)

    std_scaler = StandardScaler()
    X_scaled_train = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_scaled_test = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_scaled_train, X_scaled_test, y_train, y_test

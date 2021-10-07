import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, fft2,ifft2
from scipy.signal import welch, butter, filtfilt


def get_file_names(folder_path,returns=False):
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
    
    
    
    
    

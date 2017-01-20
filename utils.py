import os
import numpy as np
import pandas as pd

pd.options.display.max_columns = 100

import matplotlib.pyplot as plt

WORK_DIR = os.getcwd()
DATA_DIR = os.path.join(WORK_DIR, 'data')


def make_sure_path_exists(path): 
    '''
    Function to  make a new directory structure if it doesnt exists.
    Created by following: 
    # http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    Input: Directory path
    Output: Creates directories if they dont exist
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        
def getDfInfo(df):
    '''
    Function to display information about pandas data frame
    Input: Pandas data frame
    Output: 
        1. Prints the dimension of dataframe
        2. Prints null percentage in each column of dataframe    
    '''
    nrow = df.shape[0]
# print np.count_nonzero(df.isnull()) / nrow #http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe
    print ("\n***********SHAPE*************")
    print (df.shape)
    print ("\n********NULL PERCENTAGE******")
    print (df.isnull().sum() / nrow)
    df.head()
    
    
    
def change_string_to_int_list(s):
    '''
    Function to convert string to list
    Input: String
    Output: List
    '''
    
    # Remove brackets
    s = s.translate({ord(c): None for c in '\[\]'})
    # Return as list of int
    return [int(e) for e in s.split(",")]


def get_distribution(pd_series):
    '''
    Function to get distribution in Pandas series
    Input: Pandas series
    Output: Pandas series
    '''
    
    distribution = {}
    for (idx, row) in pd_series.iteritems():
        for val in row:
    #         val = str(val)
            if val in distribution:
                distribution[val] += 1
            else:
                distribution[val] = 1
                
    return pd.Series(distribution)

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
    Source: http://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

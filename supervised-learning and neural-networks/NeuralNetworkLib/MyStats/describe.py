import sys
import os
import pandas as pd
import numpy as np
from .stats import *

def iterate_data(data, func):
    ret = np.array([])
    for column in data:
        ret = np.append(ret, func(column))
    return ret


def get_numerical_data(path, header):
    i = 0
    column_labels = []
    if header == False:
        data = pd.read_csv(path, header=None)
    else:
        data = pd.read_csv(path, index_col=0)
    for (column_name, column_data) in data.iteritems():
        if isinstance(column_data[0], (int, float)) == True:
            try:
                numerical_data = np.append(numerical_data, np.array([column_data]), axis=0)
            except:
                numerical_data = np.array([column_data])
            column_labels.append(column_name)
    return (numerical_data, column_labels)

def describe(path, header=True):
    if os.path.isfile(path) == False:
        print("Error: file argument does not exist")
        exit()
    row_labels = ["count", "min", "max", "mean", "standard_deviation", "quartiles_25", "median", "quartiles_75", "mode", "skewness", "kurtosis", "missing_values"]
    functions = [count, min, max, mean, standard_deviation, quartiles_25, median, quartiles_75, mode, skewness, kurtosis, missing_values]
    numerical_data, column_labels = get_numerical_data(path, header)
    for func in functions:
        try:
            rows = np.append(rows, np.array([iterate_data(numerical_data, func)]), axis=0)
        except:
            rows = np.array([iterate_data(numerical_data, func)])
    pd.set_option('display.max_columns', None) #Force Show all columns
    print(pd.DataFrame(rows, index=row_labels, columns=column_labels))

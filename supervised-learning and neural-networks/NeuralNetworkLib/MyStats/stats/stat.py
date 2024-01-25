import numpy as np


def count(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    return column.shape[0]


def min(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    min = np.Inf
    for value in column:
        if value < min:
            min = value
    return min


def max(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    max = np.NINF
    for value in column:
        if value > max:
            max = value
    return max


def mean(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        print(column[0])
        print(type(column[0]))
        return np.nan
    try:
        column = column[~np.isnan(column)]
    except:
        pass
    res = 0
    for values in column:
        res += float(values)
    return res / column.shape[0]


def median(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    try:
        column = column[~np.isnan(column)]
    except:
        pass
    i = 0
    column = np.sort(column);
    lenght = column.shape[0]
    if lenght % 2 != 0:
        lenght /= 2
        while i < lenght:
            i += 1
        return column[i]
    else:
        lenght /= 2
        while i < lenght:
            i += 1
        return mean(np.array([column[i - 1], column[i]]))


def standard_deviation(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    try:
        column = column[~np.isnan(column)]
    except:
        pass
    res = 0
    mean_ = mean(column)
    for value in column:
        res += ((value - mean_)**2)
    res /= column.shape[0]
    return res ** 0.5


def quartiles_25(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    column = np.sort(column);
    lenght = column.shape[0]
    if lenght % 2 == 0:
        lenght //= 2
        return median(column[0:lenght])
    else:
        lenght //= 2
        return median(column[0:lenght - 1])


def quartiles_75(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    column = np.sort(column);
    lenght = column.shape[0]
    lenght //= 2
    return median(column[lenght + 1:])

# most occured value in dataset
def mode(column):
    try:
        column = column[~np.isnan(column)]
    except:
        pass
    labels = np.unique(column, return_counts=True)
    max = 0
    for i, count in enumerate(labels[1]):
        if count > max:
            max = count
            index = i
    return labels[0][index]

# If skewness is positive, the mean is bigger than the median and the distribution has a large tail of high values.
# If skewness is negative, the mean is smaller than the median and the distribution has a large tail of small values.
# Machine learning models do not function well with skewed data
def skewness(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    return (mean(column) - median(column)) / standard_deviation(column)

#Positive kurtosis indicates a thin pointed distribution.
#Negative kurtosis indicates a broad flat distribution.
#Datasets with high kurtosis have a lot of outliers, not good for ML
def kurtosis(column):
    if isinstance(column[0], (int, float, np.floating)) == False:
        return np.nan
    column = column[~np.isnan(column)]
    summation1 = 0
    summation2 = 0
    mean_ = mean(column)
    for value in column:
        summation1 += ((value - mean_)**3)
    summation1 /= column.shape[0]
    for value in column:
        summation2 += ((value - mean_)**2)
    summation2 /= column.shape[0]
    summation2 **=3
    return summation1/summation2

def missing_values(column):
    return len(column[np.isnan(column)])

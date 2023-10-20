import pandas as pd
import matplotlib
import matplotlib.pyplot as mpl
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import seaborn as sb
import os

#Avoids making same comparisons
def only_after_in_list_check(features, curr, other):
    for feature in features:
        if other == feature:
            return False
        if curr == feature:
            return True

def pair_plot(path, _class, header=True, drop_columns=None, column_range_features=None):#if header _class=name else _class is equal to y column num
    if os.path.isfile(path) == False:
        print("Error: file argument does not exist")
        exit()
    if (header == True and drop_columns == None) or (header == False and column_range_features == None):
        print("Error: function arguments")
        exit()
    if header == True:
        data = pd.read_csv(path, index_col=0)
        features = data.drop(drop_columns, axis=1)
    else:
        data = pd.read_csv(path, header=None) #use a list(range
        features = pd.read_csv(path, header=None, usecols=column_range_features) #use a list(range
    for features1 in features:
        for features2 in features:
            if only_after_in_list_check(features, features1, features2):
                sb.pairplot(data, hue=_class, vars=[features1, features2]) #sb.pairplot(data, hue="Hogwarts House") -> One huge pairplot
                mpl.show()#Seaborn plots are showed with matplotlib

def get_data(path, classes_, class_label, drop_columns):
    classes_data = []
    try:
    	data = pd.read_csv(path, index_col=0)
    except:
        print("Error: argument file")
        exit()
    data = data.drop(drop_columns, axis=1)
    for class_ in classes_:
        classes_data.append(data[data[class_label] == class_])
    return classes_data

def scatter_plot(path, classes_, class_label, drop_columns):
    len_ = len(classes_)
    if len_ > 8:
        print("Error: too much classes to show in plot")
        exit()
    data = get_data(path, classes_, class_label, drop_columns)
    features = data[0].columns[1:]
    for features1 in features:
        for features2 in features:
            if only_after_in_list_check(features, features1, features2):
                figure = data[0].plot.scatter(y=features1, x=features2, s=[2])
                for i, color in zip(range(0,len_), ["b", "g", "r", 'c', 'm', 'y', 'k', 'w'][0:len_]):
                    data[i].plot.scatter(y=features1, x=features2, s=[2], c=color, ax=figure)
                mpl.show()

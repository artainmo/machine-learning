import numpy as np
import pandas as pn
import matplotlib
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import matplotlib.pyplot as mpl

def accuracy_score(expected_values, predicted_values):
    correct = 0
    if expected_values.shape[0] == 0:
        return "Error: empty";
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if np.array_equal(expected_value, predicted_value):
            correct += 1
    return correct / expected_values.shape[0]


def positives_negatives(expected_values, predicted_values, class_=1):
    data = {"true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0}
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if np.array_equal(expected_value, predicted_value):
            if expected_value[0] != class_:
                data["true_negative"] += 1
            else:
                data["true_positive"] += 1
        else:
            if predicted_value[0] != class_:
                data["false_negative"] += 1
            else:
                data["false_positive"] += 1
    return data

def precision_score(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_positive"])

def recall_score(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_negative"])

def f1_score(expected_values, predicted_values, class_=1):
    _precision_score = precision_score(expected_values, predicted_values, class_)
    _recall_score = recall_score(expected_values, predicted_values, class_)
    return (2 * _precision_score * _recall_score) / (_precision_score + _recall_score)


def confusion_matrix(expected_values, predicted_values, class_):
    data = positives_negatives(expected_values, predicted_values, class_)
    confusion_matrix = pn.DataFrame(index=["positive class", "negative class"], columns=["positive prediction", "negative prediction"])
    confusion_matrix.at["positive class", "positive prediction"] = data["true_positive"]
    confusion_matrix.at["negative class", "positive prediction"] = data["false_positive"]
    confusion_matrix.at["positive class", "negative prediction"] = data["false_negative"]
    confusion_matrix.at["negative class", "negative prediction"] = data["true_negative"]
    return confusion_matrix


def evaluate(expected_values, predicted_values, class_=1):
    print("=============================================FEEDBACK EVALUATION===============================================")
    print("accuracy score: " + str(accuracy_score(expected_values, predicted_values)))
    print("precision score (minimize false positives): " + str(precision_score(expected_values, predicted_values, class_)))
    print("recall score (minimize false negatives): " + str(recall_score(expected_values, predicted_values, class_)))
    print("f1 score: " + str(f1_score(expected_values, predicted_values, class_)))
    print("confusion matrix:\n" + str(confusion_matrix(expected_values, predicted_values, class_)))
    print("===============================================================================================================")


def compare_different_neural_networks(list_of_neural_networks):
    input("\n==========================================\nPress Enter To Compare All Neural Networks\n==========================================")
    for NN in list_of_neural_networks:
        rem = NN.early_stopping
        NN.early_stopping = False
        NN.basic_graph()
        NN.early_stopping = rem
    mpl.legend()
    mpl.show()

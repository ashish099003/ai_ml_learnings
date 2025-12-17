import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeRegressor

def entropy(s):
    entropy = 0
    target_len = len(s)
    if target_len == 0:
        return 0

    positive_count = sum(s)
    negative_count = target_len - positive_count

    p_pos = positive_count / target_len
    p_neg = negative_count / target_len

    if p_pos > 0:
        entropy -= p_pos * np.log2(p_pos)
    if p_neg > 0:
        entropy -= p_neg * np.log2(p_neg)

    return entropy

def information_gain(parent, left_child, right_child):
    parent_entropy = entropy(parent)
    left_entropy = entropy(left_child)
    right_entropy = entropy(right_child)
    weighted_entropy = (len(left_child) / len(parent)) * left_entropy + \
                       (len(right_child) / len(parent)) * right_entropy
    return parent_entropy - weighted_entropy

def best_split(features, labels):
    best_threshold = None
    best_info_gain = -1

    for threshold in np.unique(features):
        y_left = [label for f, label in zip(features, labels) if f <= threshold]
        y_right = [label for f, label in zip(features, labels) if f > threshold]

        if len(y_left) > 0 and len(y_right) > 0:
            gain = information_gain(labels, y_left, y_right)
            if gain > best_info_gain:
                best_threshold = threshold
                best_info_gain = gain

    return best_threshold



if __name__=='__main__':

    print("Hello World")
    features = [0.58, 0.9,  0.45, 0.18, 0.5,  0.12, 0.31, 0.09, 0.24, 0.83]
    labels = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    # print(best_split(features,labels))
 
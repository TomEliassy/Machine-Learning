
"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author:
Date: May, 2018

"""

import numpy as np
from ex4_tools import DecisionStump, ListedColormap, decision_boundaries
from adaboost import AdaBoost
from decision_tree import DecisionTree
from matplotlib.pyplot import *
from bagging import Bagging

def Q3(): # AdaBoost

    val_error = list()
    train_error = list()

    X_valid = np.loadtxt(r"SynData/X_val.txt")
    X_train = np.loadtxt(r"SynData/X_train.txt")
    X_test = np.loadtxt(r"SynData/X_test.txt")
    y_train = np.loadtxt(r"SynData/y_train.txt")
    y_valid = np.loadtxt(r"SynData/y_val.txt")
    y_test = np.loadtxt(r"SynData/y_test.txt")

    rng = range(5, 105, 5)

    for T in rng:
        ab = AdaBoost(DecisionStump, T)
        ab.train(X_train, y_train)
        train_error.append(ab.error(X_train, y_train))
        val_error.append(ab.error(X_valid, y_valid))
    T = 200
    ab = AdaBoost(DecisionStump, T)
    ab.train(X_train, y_train)
    err_tr = ab.error(X_train, y_train)
    train_error.append(err_tr)
    err_vld = ab.error(X_valid, y_valid)
    val_error.append(err_vld)

    plot(list(rng) + [200], train_error)
    plot(list(rng) + [200], val_error)

    ylabel("error rate")
    xlabel("T values")

    legend(["train error", "validation error"], loc=7)
    show()

    figure(1)
    ion()
    for index,T in enumerate([1, 5, 10, 50, 100, 200]):

        ab = AdaBoost(DecisionStump, T)
        ab.train(X_train, y_train)
        subplot(2,3, index + 1)
        decision_boundaries(ab, X_train, y_train, "T = " + str(T))
     # do not close the figure

    mini = np.min(val_error)
    best_T = val_error.index(mini) * 5
    if best_T == 1000:
        best_T = 200
    print(best_T)
    ab = AdaBoost(DecisionStump, best_T)
    ab.train(X_train, y_train)
    print(ab.error(X_test, y_test))

def Q4(): # decision trees
    X_train = np.loadtxt(r"SynData/X_train.txt")
    X_test = np.loadtxt(r"SynData/X_test.txt")
    X_valid = np.loadtxt(r"SynData/X_val.txt")
    y_valid = np.loadtxt(r"SynData/y_val.txt")
    y_train = np.loadtxt(r"SynData/y_train.txt")
    y_test = np.loadtxt(r"SynData/y_test.txt")
    D = [3, 6, 8, 10, 12]
    val_error = list()
    train_error = []

    for d in D:
        dt = DecisionTree(d)
        dt.train(X_train, y_train)
        err = dt.error(X_train, y_train)
        train_error.append(err)
        val_error.append(dt.error(X_valid, y_valid))
    plot(D, train_error)
    plot(D, val_error)
    xlabel("d")
    ylabel("error rate")
    legend(["train error", "validation error"], loc=1)
    show()
    for index, d in enumerate(D):
        dt = DecisionTree(d)
        dt.train(X_train, y_train)
        subplot(2,3, index + 1)
        str_d = str(d)
        decision_boundaries(dt, X_train, y_train, "d = " + str_d)
    # do not close the figure
    best_d = D[val_error.index(np.min(val_error))]
    print(best_d)
    dt = DecisionTree(best_d)
    dt.train(X_train, y_train)
    print(dt.error(X_test, y_test))

    # Bonus:
    val_error = list()
    for B in range(5,105,5):
        print("B: " + str(B))
        bag = Bagging(DecisionTree, B, best_d)
        bag.train(X_train, y_train)
        val_error.append(bag.error(X_valid, y_valid))

    plot(range(5,105,5), val_error)
    xlabel("B")
    ylabel("validation error rate")
    show()
    best_b = list(range(5,105,5))[val_error.index(np.min(val_error)) + 5]
    print(best_b)
    bag = Bagging(DecisionTree, best_b, best_d)
    bag.train(X_train, y_train)
    print(bag.error(X_test, y_test))


def Q5(): # spam data
    n_folds = 5
    # creating the data
    y = np.loadtxt(r"SpamData/spam.data", usecols=-1)
    y[y == 0] = -1
    data = np.loadtxt(r"SpamData/spam.data")
    X = data[:,:-1]

    npa = np.array(range(len(X)))
    sample_data_idx = np.random.permutation(npa)
    X_test, y_test, X_train, y_train = X[sample_data_idx[:1536]], y[sample_data_idx[:1536]], \
                                       X[sample_data_idx[1536:]], y[sample_data_idx[1536:]]
    T_values = [5, 50, 100, 200, 500, 1000]
    d_values = [5, 8, 10, 12, 15, 18]

    # splitting to folds
    rng = range(len(X_train))
    cross_val_idx = np.random.permutation(np.array(rng))

    temp1 = []
    temp2 = []
    for i in range(n_folds):
        idx = cross_val_idx[int(i * len(X_train) / n_folds): int((i + 1) * len(X_train) / n_folds)]
        temp1.append(X_train[idx])
        temp2.append(y_train[idx])

    folds = np.array(temp1)
    folds_y = np.array(temp2)

    AB_sd = list()
    DT_sd = list()
    AB_mean_errors = list()
    DT_mean_errors = list()

    # cross validation:
    for val in range(len(T_values)):
        print("val" + str(val))
        # the len of T and d is the same
        ab_errors = list()
        dt_errors = list()
        for fold_idx in range(n_folds):
            print("fold idx " + str(fold_idx))
            fold_test = folds[fold_idx]
            fold_test_y = folds_y[fold_idx]
            other_fold_idx = list()
            for i in range(n_folds):
                if i!=fold_idx:
                    other_fold_idx.append(i)
            other_folds = np.concatenate(folds[other_fold_idx])
            other_folds_y = np.concatenate(folds_y[other_fold_idx])

            # adaboost training on current fold:
            ab = AdaBoost(DecisionStump, T_values[val])
            ab.train(other_folds, other_folds_y)
            ab_errors.append(ab.error(fold_test, fold_test_y))

            # DT training on current fold:
            dt = DecisionTree(d_values[val])
            dt.train(other_folds, other_folds_y)
            dt_errors.append(dt.error(fold_test, fold_test_y))

        # add the mean errors:
        np_arr = np.array(dt_errors).mean()
        DT_mean_errors.append(np_arr)
        DT_sd.append(np.std(dt_errors))
        AB_mean_errors.append(np.array(ab_errors).mean())
        AB_sd.append(np.std(ab_errors))


    # plotting the errors in order to view the best T and d parameters:
    errorbar(T_values, AB_mean_errors, np.reshape(np.array(AB_sd), (len(T_values), 1)), ecolor="blue")
    xlabel("T values")
    ylabel("avg error rate")
    show()
    errorbar(d_values, DT_mean_errors, np.reshape(np.array(DT_sd), (len(T_values), 1)), ecolor="blue")
    xlabel("D")
    ylabel("avg error rate")
    show()
    # best_b = AB_mean_errors.index(np.min(AB_mean_errors))
    best_b = 200
    print("adaboost best:" + str(best_b))
    ab = AdaBoost(DecisionStump, best_b)
    ab.train(X_train, y_train)
    print(ab.error(X_test, y_test))
    # best_b = DT_mean_errors.index(np.min(DT_mean_errors))
    best_b = 10
    print("DT best parameter:" + str(best_b))
    dt = DecisionTree(best_b)
    dt.train(X_train, y_train)
    print(dt.error(X_test, y_test))
    return

def main():
    Q3()
    Q4()
    Q5()
if __name__ == '__main__':
    main()

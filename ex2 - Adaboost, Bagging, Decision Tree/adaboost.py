"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m, d = X.shape
        D = np.full(m, 1/m)
        for i in range(self.T):
            self.h[i] = self.WL(D, X, y)
            error_i = np.dot(D, (y != self.h[i].predict(X)))
            self.w[i] = 0.5 * np.log((1/error_i) - 1)
            y_hat = self.h[i].predict(X)
            D = (D * np.exp(-self.w[i] * y * y_hat)) / (D * np.exp(-self.w[i] * y * y_hat)).sum()

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        temp = []
        for t in range(self.T):
            temp.append(np.dot(self.w[t], self.h[t].predict(X)))
        return np.sign(np.array(temp).sum(axis=0))

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y[i] != y_hat[i] for i in range(len(y)))/len(y)


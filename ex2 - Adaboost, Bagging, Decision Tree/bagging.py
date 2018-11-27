"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for classification with Bagging.

Author: Yoav Wald

"""
import numpy as np

class Bagging(object):

    def __init__(self, L, B, size_T):
        """
        Parameters
        ----------
        L : the class of the base learner
        T : the number of base learners to learn
        """
        self.L = L
        self.B = B
        self.size_T = size_T # the parameter for the learner
        self.h = [None]*B     # list of base learners

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m, d = X.shape
        for b in range(self.B):
            S_tag_idx = np.random.choice(np.array(range(m)), m)
            S_tag = X[S_tag_idx]
            y_tag = y[S_tag_idx]
            self.h[b] = self.L(self.size_T)
            self.h[b].train(S_tag, y_tag)

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        temp = 0
        for i in range(self.B):
            temp += (1/self.B)* self.h[i].predict(X)

        return np.sign(temp)


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        sum = 0
        for i in range(len(y)):
            if y[i] != y_hat[i]:
                sum += 1
        return sum / len(y)
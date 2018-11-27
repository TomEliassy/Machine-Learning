"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: CART

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,
                 misclassification = 1,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.label = label
        self.misclassification = misclassification


class DecisionTree(object):
    """ A decision tree for binary classification.
        max_depth - the maximum depth allowed for a node in this tree.
        Training method: CART
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        temp = list()
        for i in range(X.shape[1]):
            temp.append(np.unique(X[:,i]))
        A = np.array(temp)

        self.root = self.CART(X, y, A, 0)


    def calc_errors(X, y, thres, dim):
        errors = 0
        for sample_idx in range(len(X)):
            if X[sample_idx][dim] <= thres:
                if y[sample_idx] != 1:
                    errors = errors + 1
            elif y[sample_idx] != -1:
                errors = errors + 1
        return errors / len(X)

    def split_tree(self,X, y, A):

        min_label = 1
        min_error = 1
        min_threshold = 0
        min_dim = 0

        for dim in range(len(A)):
            for threshold in A[dim]:
                curr_error_1 = DecisionTree.calc_errors(X, y, threshold, dim)
                curr_error_minus1 = 1 - curr_error_1
                if curr_error_1 < min_error:
                    min_dim = dim
                    min_label = 1
                    min_error = curr_error_1
                    min_threshold = threshold

                if curr_error_minus1 < min_error:
                    min_dim = dim
                    min_label = -1
                    min_error = curr_error_minus1
                    min_threshold = threshold


        return Node(False, samples=0, feature=min_dim, theta=min_threshold, label=min_label,
                    misclassification=min_error)

    def CART(self,X, y, A, depth):

        if depth == self.max_depth:
            leaf = self.split_tree(X, y, A)
            leaf.leaf = True
            return leaf


        new_node = self.split_tree(X, y, A)
        if new_node.misclassification == 0:
            new_node.leaf = True
            return new_node
        feat = new_node.feature
        thet = new_node.theta
        X_less = X[X[:, feat] <= thet]
        y_less = y[X[:, feat] <= thet]
        rng_l = range(X_less.shape[1])
        if len(X_less) < 1:
            new_node.left = Node(leaf=True, label=new_node.label)
        else:
            A_less = np.array(np.array([np.unique(X_less[:,i]) for i in rng_l]))
            new_node.left = self.CART(X_less, y_less, A_less, depth + 1)
        X_great = X[X[:, feat] > thet]
        y_great = y[X[:, feat] > thet]
        rng_g = range(X_great.shape[1])
        if len(X_great) < 1:
            new_node.right = Node(leaf=True, label=-new_node.label)
        else:
            A_great = np.array(np.array([np.unique(X_great[:,i]) for i in rng_g]))
            new_node.right = self.CART(X_great, y_great, A_great, depth + 1)
        return new_node



    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y_pred = list()
        for x in X:
            curr_node = self.root
            while not curr_node.leaf:
                if x[curr_node.feature] > curr_node.theta:
                    curr_node = curr_node.right
                else:
                    curr_node = curr_node.left

            if curr_node != self.root:

                y_pred.append(curr_node.label)
            else:
                if x[curr_node.feature] <= self.root.theta:
                    y_pred.append(curr_node.label)
                else:
                    y_pred.append(-curr_node.label)
        return np.array(y_pred)

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y[i] != y_hat[i] for i in range(len(y))) / len(y)

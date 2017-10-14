"""Module with different fitness functions implemented to be used by the CRO algorithm.

The functions' only argument must be an individual (coral) and return its fitness, a number.
The fitness might require other arguments, in that case the partial function in python's
functools module is a very good option
"""
from __future__ import division
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

def max_ones(coral):
    """Returns the percentage of 1's in the coral.

    This function assumes 'coral' is a list, it could be further improved if it was a numpy
    array
    """
    return 100*(sum(coral) / len(coral))

def feature_selection(coral, Xt, yt, clf, metric=roc_auc_score, random_seed=None):
    """Returns the fitness (given by metric) of the selected features given by coral,
    when using Xt and yt for training the model clf
    """
    # offset % of data for training, the rest for testing
    offset = int(Xt.shape[0] * 0.9)

    X, y = shuffle(Xt, yt, random_state=random_seed)
    X = np.multiply(X, coral)
   
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    # train model
    clf.fit(X_train, y_train)   

    # Compute metric
    fitness = metric(y_test, clf.predict_proba(X_test))

    return fitness

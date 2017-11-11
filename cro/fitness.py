from __future__ import division

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import *

"""
Module with different fitness functions implemented to be used by the CRO algorithm.

The functions' only argument must be an individual (coral) and return its fitness, a number.
The fitness might require other arguments, in that case the partial function in python's functools module is a very good option
"""

def max_ones(coral):
    """
    Description: Returns the percentage of 1's in the coral. This function assumes 'coral' is a list,
    it could be further improved if it was a numpy array

    Input:
        - coral
    Output:
        - fitness
    """
    return 100*(sum(coral) / len(coral))

def feature_selection(coral, X, y, model, 
                      get_prediction = lambda model, X: model.predict(X),
                      metric=roc_auc_score, random_seed=None):   
    """
    Description: Returns the fitness (given by metric) of the selected features given by coral,
    when using Xt and yt for training the model clf

    Input:
        - coral : an individual
        - X: Data input
        - y: Data output
        - model: instance of the model to be trained
        - get_prediction: function that accepts the model and X and outputs the vector 
                          that will be used in the metric (predictions, scores...)
        - metric: metric that will be used as fitness
    Output:
        - fitness
    """
    # offset % of data for training, the rest for testing
    offset = int(X.shape[0] * 0.9)

    Xs, ys = shuffle(X, y, random_state=random_seed)
    Xs = np.multiply(Xs, coral) 
    
    X_train, y_train = Xs[:offset], ys[:offset]
    X_test, y_test   = Xs[offset:], ys[offset:]

    # train model
    model.fit(X_train, y_train)   

    # Compute metric
    y_pred = get_prediction(model, X_test)
    fitness = metric(y_test, y_pred)

    return fitness

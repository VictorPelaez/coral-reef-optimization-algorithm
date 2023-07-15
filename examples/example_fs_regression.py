#!/usr/bin/env python
# coding=utf-8
###############################################################################

import context
from cro.cro import CRO
from cro.fitness import feature_selection
from cro.report import plot_results

import time
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == '__main__':
    """
    Example II: feature selection, regression (min mse) 
    """
    
    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 30                  # Number of generations
    N  = 10                    # MxN: reef size
    M  = 10                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.2                   # Depredation prob.
    opt= 'min'                 # flag: 'max' for maximizing and 'min' for minimizing
    ## ------------------------------------------------------

    L = 100 # number of features
    k_fs = 20

    # generate regression dataset
    X, y = make_regression(n_samples=2000, n_features=L, n_informative=k_fs, noise=0.1, random_state=1)
    
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k="all")
    # learn relationship from training data
    fs.fit(X, y)
    arr = np.array(fs.scores_)
    kbest_index = arr.argsort()[-k_fs:][::-1]
    print("KBest method feature selection:", kbest_index) # [40 92 60  9 39 34 28 46]

    lr = LinearRegression()
    lr.fit(X, y)
    # evaluate the model
    yhat = lr.predict(X)
    # evaluate predictions
    mse = mean_squared_error(y, yhat)
    print('MSE: %.3f' % mse)
    

    # CRO comparasion

    fitness_coral = partial(feature_selection, X=X, y=y, model=lr,
                            get_prediction=lambda lr, X: lr.predict(X), 
                            metric=mean_squared_error)
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L, seed=13, verbose=True, n_jobs=1)
    cro.fit(X, y, lr)
    REEF, REEFpob, REEFfitness, ind_best, Fitness = cro.get_results()
    print("Example II: feature selection, regression (min mse): ", time.time() - start, "seconds.")

    plot_results(cro, filename=None)

    cro_index = np.where(REEFpob[ind_best, :]==1)
    print(sum(REEFpob[ind_best, :]))

    same_fs = []
    for kb in kbest_index:
        same_fs.append(kb in cro_index[0])
    print("Porcentage same features", str(100*sum(same_fs)/k_fs), "%")    





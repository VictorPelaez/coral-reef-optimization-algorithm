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
    Ngen = 6                  # Number of generations
    N  = 12                    # MxN: reef size
    M  = 12                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.2                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.3                   # Depredation prob.
    opt = 'min'                 # flag: 'max' for maximizing and 'min' for minimizing
    ## ------------------------------------------------------

    L = 200 # Number of features
    k_fs = 2

    # generate regression dataset
    X, y = make_regression(n_samples=2000, n_features=L, n_informative=k_fs, noise=0.8, random_state=13)
    
    print("\n------------------------ KBest Scikit + Linear regression ------------------------")
    # Configure to select all features
    fs = SelectKBest(score_func=f_regression, k="all")
    fs.fit(X, y)
    arr = np.array(fs.scores_)
    kbest_index = arr.argsort()[-k_fs:][::-1]
    print("KBest method feature selection:", kbest_index) # [54, 105]

    lr = LinearRegression()
    lr.fit(X, y)
    yhat = lr.predict(X)
    mse = mean_squared_error(y, yhat)  # evaluate predictions
    print('MSE: %.6f' % mse)
    print("------------------------")

    

    #------------------- CRO comparasion
    print("\n------------------------ CRO running ------------------------")
    fitness_coral = partial(feature_selection, X=X, y=y, model=lr,
                            get_prediction=lambda lr, X: lr.predict(X), 
                            metric=mean_squared_error, random_seed=13)
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L, seed=13, verbose=True)
    cro.fit(X, y, lr)
    REEF, REEFpob, REEFfitness, ind_best, Fitness = cro.get_results()
    plot_results(cro, filename=None)
    print("Example II: feature selection, regression (min mse): ", time.time() - start, "seconds.") 
    print("------------------------")

    print("\n------------------------ KBest approach ------------------------")
    kb_corals = int(np.round(0.9*np.shape(REEFfitness)[0]))
    result = np.argpartition(REEFfitness, kb_corals)
    kbest_ind = result[:kb_corals]
    # print('Best K corals: ', REEFpob[kbest_ind, :].sum(axis=0)/kb_corals)
    print(np.where(REEFpob[kbest_ind, :].sum(axis=0)/kb_corals==1))
    print("------------------------")



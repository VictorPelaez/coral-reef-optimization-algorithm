#!/usr/bin/env python
# coding=utf-8
###############################################################################

import context
from cro.cro import CRO
from cro.fitness import hyperparameter_selection
from cro.utils import load_data
from cro.report import plot_results

import time
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn import datasets, ensemble
import numpy as np

if __name__ == '__main__':
    
    """
    Example I: hyper-parameter selection
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    """
    
    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 2                   # Number of generations
    N  = 20                    # MxN: reef size
    M  = 10                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.7                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
    mode = 'cont'
    mutation = {'shrink': 1.}
    grid = {'learning_rate': [0.1, 1.]}
    ## ------------------------------------------------------
    
    dataset = datasets.load_boston()
    L = 20
    X = dataset.data
    y = dataset.target
    
    params = {'n_estimators': 60, 'max_depth': 4, 'min_samples_split': 2}
    gbr = ensemble.GradientBoostingRegressor(**params)  
    
    fitness_coral = partial(hyperparameter_selection, X=X, y=y, model=gbr,
                            get_prediction=lambda gbr, X: gbr.predict(X), 
                            metric=r2_score, param='learning_rate')
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L,
              mode=mode, mutation=mutation, param_grid=grid, seed=13, verbose=True)
    
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit(X, y, gbr)
    print("Example I: hyper-parameter selection, regression (max r2): ", time.time() - start, "seconds.")

    plot_results(Bestfitness, Meanfitness, cro, filename=None)
    print(np.max(REEFpob[ind_best, :]))
    
    
    """
    Example II: hyper-parameter selection     
    """

    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 10                  # Number of generations
    N  = 10                    # MxN: reef size
    M  = 10                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
    mode = 'disc'
    grid = {'n_neighbors': [1, 10]}
    ## ------------------------------------------------------
    
    dataset = load_data('voice')
    L = len(grid)
    X = dataset.data
    y = dataset.target
    
    clf = KNeighborsClassifier()
    
    fitness_coral = partial(hyperparameter_selection, X=X, y=y, model=clf,
                            get_prediction = lambda clf, X: clf.predict_proba(X)[:, 1], 
                            metric=roc_auc_score, param='n_neighbors', random_seed=13)
    
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L,
              mode=mode, param_grid=grid, seed=13, verbose=True)
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit(X, y, clf)

    plot_results(Bestfitness, Meanfitness, cro, filename=None)
    print("Example II: hyper-parameter selection: ", time.time() - start, "seconds.")
    
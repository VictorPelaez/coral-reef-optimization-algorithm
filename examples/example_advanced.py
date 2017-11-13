#!/usr/bin/env python
# coding=utf-8
###############################################################################

import context
from cro.cro import CRO
from cro.fitness import feature_selection
from cro.utils import load_data
from cro.report import plot_results

import time
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn import datasets, ensemble
import numpy as np

if __name__ == '__main__':
    
    """
    Example I: feature selection Classification (max auc)
     
    https://www.kaggle.com/primaryobjects/voicegender
    This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech.
    The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. It contains 20 features and I added 10 noisy!
    """

    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 20                  # Number of generations
    N  = 10                    # MxN: reef size
    M  = 10                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
    ## ------------------------------------------------------
    
    dataset = load_data('voice')
    L = dataset.data.shape[1] # number of features
    X = dataset.data
    y = dataset.target
    
    clf = KNeighborsClassifier(2)
    
    fitness_coral = partial(feature_selection, X=X, y=y, model=clf,
                            get_prediction = lambda clf, X: clf.predict_proba(X)[:, 1], 
                            metric=roc_auc_score)
    
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L, seed=13, verbose=True)
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit(X, y, clf)

    plot_results(Bestfitness, Meanfitness, cro, filename=None)
    print("Example I: feature selection Classification (max auc): ", time.time() - start, "seconds.")
    
    names = np.array(dataset.feature_names)
    print(names[REEFpob[ind_best, :]>0])
    
    """
    Example II: feature selection, regression (min mse) 
    """
    
    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 25                  # Number of generations
    N  = 10                    # MxN: reef size
    M  = 10                    # MxN: reef size
    Fb = 0.8                   # Broadcast prob.
    Fa = 0.2                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.7                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'min'                 # flag: 'max' for maximizing and 'min' for minimizing
    
    ## ------------------------------------------------------
    
    dataset = datasets.load_boston()
    L = dataset.data.shape[1] # number of features
    X = dataset.data
    y = dataset.target
    
    params = {'n_estimators': 60, 'max_depth': 4, 'min_samples_split': 2}
    gbr = ensemble.GradientBoostingRegressor(**params)  
    
    fitness_coral = partial(feature_selection, X=X, y=y, model=gbr,
                            get_prediction=lambda gbr, X: gbr.predict(X), 
                            metric=mean_squared_error)
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, fitness_coral, opt, L, seed=13, verbose=True)
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit(X, y, gbr)
    print("Example II: feature selection, regression (min mse): ", time.time() - start, "seconds.")

    plot_results(Bestfitness, Meanfitness, cro, filename=None)
    
    names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    print(names[REEFpob[ind_best, :]>0])

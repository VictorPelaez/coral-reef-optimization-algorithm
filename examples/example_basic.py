#!/usr/bin/env python
# coding=utf-8
###############################################################################

import context
from cro.cro import CRO
from cro.fitness import max_ones
from cro.report import plot_results

import time

if __name__ == '__main__':

    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 30                  # Number of generations
    N  = 20                    # MxN: reef size
    M  = 20                    # MxN: reef size
    Fb = 0.7                   # Broadcast prob.
    Fa = 0.1                   # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
    npolyps = 3                # Number of polyps to be mutated in the brooding operator

    L = 100
    ke = 0.2
    ## ------------------------------------------------------

    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, max_ones, opt, L, verbose=True, ke=ke, npolyps=npolyps)
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit()
    print("Example I: max_ones problem", time.time() - start, "seconds.")

    info = dict(N=N, M=M, Fb=Fb, Fa=Fa, Fd=Fd, r0=r0, k=k, L=L, Pd=Pd)
    plot_results(Bestfitness, Meanfitness, title_info=info, filename=None)
    
    ## ------------------------------------------------------
    ## Parameters initialization
    ## ------------------------------------------------------
    Ngen = 30                  # Number of generations
    N  = 30                    # MxN: reef size
    M  = 30                    # MxN: reef size
    Fb = 0.85                  # Broadcast prob.
    Fa = 0.05                  # Asexual reproduction prob.
    Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
    r0 = 0.6                   # Free/total initial proportion
    k  = 3                     # Number of opportunities for a new coral to settle in the reef
    Pd = 0.1                   # Depredation prob.
    opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
    npolyps = 5                # Number of polyps to be mutated in the brooding operator
    
    L = 20
    ke = 0.2
    mode = 'disc'
    grid = {'x': [2, 10]}      # Discrete values between 2 and 10
    ## ------------------------------------------------------
    
    start = time.time()
    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, max_ones, opt, L, verbose=False, ke=ke, npolyps=npolyps, mode=mode, param_grid=grid)
    (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit()
    print("Example II: max problem in a discrete interval", time.time() - start, "seconds.")

    info = dict(N=N, M=M, Fb=Fb, Fa=Fa, Fd=Fd, r0=r0, k=k, L=L, Pd=Pd)
    plot_results(Bestfitness, Meanfitness, title_info=info, filename=None)

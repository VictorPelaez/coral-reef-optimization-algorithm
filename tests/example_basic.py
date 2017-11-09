# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cro.cro import CRO
from cro.fitness import max_ones

import time

if __name__ == '__main__':
    print(CRO)
    print(max_ones)
    

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
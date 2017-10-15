#!/usr/bin/env python
# coding=utf-8
###############################################################################

import os
import sys
from cro import *

def test_croCreation():
    """
    Test that the REEF has a proper shape
    """
    L = 8
    N = 2
    M = 2
    fitness_coral = lambda coral: 1 # Dummy fitness
    cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L)
    (REEF, REEFpob) = cro.reefinitialization()
    assert REEFpob.shape == (N*M, L)

def test_croInit():
    """
    Test that the number of corals in the reef is greater than 0
    """
    L = 8
    N = 2
    M = 2
    fitness_coral = lambda coral: 1 # Dummy fitness
    cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L)
    (REEF, REEFpob) = cro.reefinitialization()
    assert len(np.where(REEF!=0)[0]) > 0

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
    

def test_reefinitializationDisc():
    """
    Test that reefinitialization function works in a discrete mode option, any coral lower than required min value and
    any higher than max 
    """
    L = 8
    N = 2
    M = 2
    grid = {'x': [1, 10]}      # Discrete values between 2 and 10
    fitness_coral = lambda coral: 1 # Dummy fitness
    
    cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L, mode='disc', param_grid=grid)
    (REEF, REEFpob) = cro.reefinitialization()
    p = sum(REEFpob[np.where(REEFpob!=0)]<grid['x'][0]) + sum(REEFpob[np.where(REEFpob!=0)]>grid['x'][1])
    assert p == 0
    
    

    

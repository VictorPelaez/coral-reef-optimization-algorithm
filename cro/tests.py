#!/usr/bin/env python
# coding=utf-8
###############################################################################

import numpy as np
from cro import CRO
from .larvaemutation import get_larvaemutation_function



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
    grid = {'x': [2, 10]}      # Discrete values between 2 and 10
    fitness_coral = lambda coral: 1 # Dummy fitness
    
    cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L, mode='disc', param_grid=grid)
    (REEF, REEFpob) = cro.reefinitialization()
    p = sum(REEFpob[np.where(REEFpob!=0)]<grid['x'][0]) + sum(REEFpob[np.where(REEFpob!=0)]>grid['x'][1])
    assert p == 0

def test_larvaesettling_emptyreef():
    """
    """
    REEFpob = np.array([[0,0,0,0],
                        [0,0,0,0], 
                        [0,0,0,0],
                        [0,0,0,0]])
    
    REEF = np.array([0,0,0,0])
    REEFfitness = np.array([0,0,0,0])

    larvae = np.array([[1,0,0,0],
                       [0,1,0,0], 
                       [0,0,1,0],
                       [0,0,0,1]])
    
    larvaefitness = np.array([1,1,1,1])

    N, L = REEFpob.shape
    fitness_coral = lambda coral: 1 # Dummy fitness
    cro = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L)

    REEF_res, REEFpob_res, REEFfitness_res = cro.larvaesettling(REEF, REEFpob, REEFfitness,
                                                                larvae, larvaefitness)

    np.testing.assert_almost_equal(REEF_res, np.array([1,1,1,1]))
    np.testing.assert_almost_equal(REEFpob_res, larvae)
    np.testing.assert_almost_equal(REEFfitness_res, larvaefitness)

def test_larvaesettling_nonemptyreef():
    """
    """
    REEFpob = np.array([[0,0,0,0],
                        [0,0,0,1], 
                        [0,0,1,0],
                        [1,0,1,1]])
    
    REEF = np.array([0,1,1,1])
    REEFfitness = -np.array([0,1,2,11])

    larvae = np.array([[1,0,0,0],
                       [0,1,1,0], 
                       [0,1,0,0],
                       [1,0,0,1]])
    
    larvaefitness = -np.array([8,6,4,9])

    N, L = REEFpob.shape
    fitness_coral = lambda coral: 1 # Dummy fitness
    cro = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L, seed=0)

    REEF_res, REEFpob_res, REEFfitness_res = cro.larvaesettling(REEF, REEFpob, REEFfitness,
                                                                larvae, larvaefitness)
    """
    Due to the passed seed,
    [1,0,0,0] will be placed in the empty coral (index 0)

    Then, larva [0,1,1,0] will try to settle in indices [0,3,1], settling in the third try (index 1)
    Larva [0,1,0,0] will try in indices [0,3,3], being discarded
    Larva [1,0,0,1] will try in indices [3,3,1], settling in the third try (index 1)
    
    Thus, the whole REEF will be populated after the settling, with a population:
    [[1,0,0,0], and a fitness [[8,
     [1,0,0,1],                 9,
     [0,0,1,0],                 2,
     [1,0,1,1]]                 11]]
    """
    REEFpob_exp = np.array([[1,0,0,0],
                            [1,0,0,1],
                            [0,0,1,0],
                            [1,0,1,1]])
    REEFfitness_exp = -np.array([8,9,2,11])

    np.testing.assert_almost_equal(REEF_res, np.array([1,1,1,1]))
    np.testing.assert_almost_equal(REEFpob_res, REEFpob_exp)
    np.testing.assert_almost_equal(REEFfitness_res, REEFfitness_exp)


def test_larvaemutattion():
    """
    Test mutated larvae in a given position, discrete mode
    """
    
    larvae = np.array([[2, 4, 4, 9, 10, 8, 3, 9],
                       [2, 9, 6, 7, 6, 5, 8, 3],
                       [3, 7, 8, 10, 6, 7, 8, 8]])
    
    pos = np.array([[0, 3, 5]])
    mode = 'disc'
    grid = {'x': [2, 10]}      # Discrete values between 2 and 10
   
    #fitness_coral = lambda coral: 1 # Dummy fitness
    #cro = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
    #          fitness_coral=fitness_coral, opt='max', L=10, seed=0, mode=mode, param_grid=grid)
    
    larvaemutation_function = get_larvaemutation_function(mode)
    larvaemutated = larvaemutation_function(larvae, pos, delta=1, param_grid=grid)
    
    #larvaemutated = cro._larvaemutation(larvae, pos)
    goodsol = np.array([[6, 4, 4, 9, 10, 8, 3, 9],
                        [2, 9, 6, 5, 6, 5, 8, 3],
                        [3, 7, 8, 10, 6, 5, 8, 8]])

    np.testing.assert_almost_equal(larvaemutated, goodsol)   

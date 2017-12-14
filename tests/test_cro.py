#!/usr/bin/env python
# coding=utf-8
###############################################################################
# run with:
# python -m pytest tests/test_cro.py
import numpy as np

from cro.cro import CRO

# ------------------------------------------------------
# cro module
# ------------------------------------------------------

# GENERIC VARIABLES FOR TESTING
L = 8
N = 2
M = 2
grid = {'x': [2, 10]} # Discrete values between 2 and 10

fitness_coral = lambda coral: 1 # Dummy fitness
cro_disc = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
          fitness_coral=fitness_coral, opt='max', L=L, seed=13, mode='disc', param_grid=grid)

cro_bin = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
          fitness_coral=fitness_coral, opt='max', L=L, seed=13)


def test_croCreation():
    """
    Test that the REEF has a proper shape
    TARGET: cro -> reefinitialization
    """
    (_, REEFpob_res) = cro_bin.reefinitialization()
    assert REEFpob_res.shape == (N*M, L)
    (_, REEFpob_res) = cro_disc.reefinitialization()
    assert REEFpob_res.shape == (N*M, L)


def test_croInit():
    """
    Test that the number of corals in the reef is greater than 0
    TARGET: cro -> reefinitialization
    """
    (REEF_res, _) = cro_bin.reefinitialization()
    assert len(np.where(REEF_res!=0)[0]) > 0
    (REEF_res, _) = cro_disc.reefinitialization()
    assert len(np.where(REEF_res!=0)[0]) > 0


def test_reefinitializationDisc():
    """
    Test that reefinitialization function works in a discrete mode option, any coral lower than required min value and
    any higher than max 
    TARGET: cro -> reefinitialization
    """
    (_, REEFpob_res) = cro_disc.reefinitialization()
    p = sum(REEFpob_res[np.where(REEFpob_res!=0)]<grid['x'][0]) + sum(REEFpob_res[np.where(REEFpob_res!=0)]>grid['x'][1])
    assert p == 0


def test_fitness():
    """
    Test that the fitness is computed correctly.
    Check that, when the CRO's opt parameter is "max", the fitness function
        actually computes -fitness
    """
    fitness_coral = lambda coral: coral.sum()
    REEFpob_test = np.array([[1,1,1,1],
                             [1,0,1,1],
                             [1,0,0,1],
                             [0,1,0,0]])

    cro_min = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='min', L=L, seed=13)
    fitness_min_expected = np.array([4,3,2,1])

    fitness_min = cro_min.fitness(REEFpob_test)
    np.testing.assert_array_equal(fitness_min, fitness_min_expected)

    # When maximizing, internally we actually minimize -fitness function
    cro_max = CRO(Ngen=10, N=2, M=2, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
              fitness_coral=fitness_coral, opt='max', L=L, seed=13)
    fitness_max_expected = -np.array([4,3,2,1])

    fitness_max = cro_max.fitness(REEFpob_test)
    np.testing.assert_array_equal(fitness_max, fitness_max_expected)


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

    REEF_res, REEFpob_res, REEFfitness_res = cro_disc.larvaesettling(REEF, REEFpob, REEFfitness,
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
 
    
def test_brooding(): 
    """
    Test brooding function in cro module
    TARGET: cro -> brooding
    """
    
    REEFpob = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1]])
        
    REEF = np.array((REEFpob.any(axis=1)),int)
    brooders = cro_bin.brooding(REEF, REEFpob)
    np.testing.assert_almost_equal(brooders, np.array([[1, 0, 0, 0, 1, 1, 1, 1]]))
    
    
def test_settle_larvae():
    """
    Test Settle the given larvae in the REEF in the given indices
    TARGET: cro -> _settle_larvae
    """
    
    REEFpob = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1]])
        
    REEF = np.array((REEFpob.any(axis=1)),int)
    indices = 0
    larvae = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    REEFfitness = np.array([0,1,0])
    larvaefitness = 1
         
    (REEF_res, REEFpob_res, REEFfitness_res) = cro_bin._settle_larvae(larvae, larvaefitness, REEF, REEFpob, REEFfitness, indices)
    
    assert REEF_res[indices] == 1
    np.testing.assert_almost_equal(REEFpob_res[indices, :], larvae)
    np.testing.assert_almost_equal(REEFfitness_res[indices], larvaefitness)

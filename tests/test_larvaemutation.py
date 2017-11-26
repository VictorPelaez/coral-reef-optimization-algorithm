#!/usr/bin/env python
# coding=utf-8
###############################################################################
# run with:
# python -m pytest tests/test_larvaemutation.py
import numpy as np

from cro.larvaemutation import get_larvaemutation_function

# ------------------------------------------------------
# larvaemutation module
# ------------------------------------------------------

def test_bin_larvaemutattion():
    """
    Test mutated larvae in a given position, binary mode
    TARGET: larvaemutation -> bin_larvaemutation
    """
    
    larvae = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1]])
    
    pos = np.array([[0, 3, 5]])
    mode = 'bin'
   
    larvaemutation_function = get_larvaemutation_function(mode)
    larvaemutated = larvaemutation_function(larvae, pos, seed=13)
    
    goodsol = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 1, 1]])

    np.testing.assert_almost_equal(larvaemutated, goodsol) 
    
def test_disc_larvaemutattion():
    """
    Test mutated larvae in a given position, discrete mode
    TARGET: larvaemutation -> disc_larvaemutation
    """
    
    larvae = np.array([[2, 4, 4, 9, 10, 8, 3, 9],
                       [2, 9, 6, 7, 6, 5, 8, 3],
                       [3, 7, 8, 10, 6, 7, 8, 8]])
    
    pos = np.array([[0, 3, 5]])
    mode = 'disc'
    grid = {'x': [2, 10]}      # Discrete values between 2 and 10
    
    larvaemutation_function = get_larvaemutation_function(mode)
    larvaemutated = larvaemutation_function(larvae, pos, delta=1, param_grid=grid, seed=13)
    
    goodsol = np.array([[5, 4, 4, 9, 10, 8, 3, 9],
                        [2, 9, 6, 6, 6, 5, 8, 3],
                        [3, 7, 8, 10, 6, 6, 8, 8]])

    np.testing.assert_almost_equal(larvaemutated, goodsol)   


def test_cont_simple_larvaemutattion():
    """
    Test mutated larvae in a given position, cont mode
    mutation type: simple gaussian + correction
    TARGET: larvaemutation -> cont_larvaemutation
    """ 
    
    larvae = np.array([[2.1, 4.2, 4.3, 9.4, 9.9, 8.6, 3.7, 9.8],
                       [2.1, 9.2, 6.3, 7.4, 6.5, 5.6, 8.7, 3.8],
                       [3.1, 7.2, 8.3, 9.4, 6.5, 7.6, 8.7, 8.8]])
    
    pos = np.array([[0, 3, 5]])
    mode = 'cont'
    mut_type = 'simple'
    grid = {'x': [2., 10.]}      # Discrete values between 2 and 10
    
    larvaemutation_function = get_larvaemutation_function(mode)
    larvaemutated = larvaemutation_function(larvae, pos, param_grid=grid, mut_type=mut_type, seed=13)
    
    goodsol = np.array([[2., 4.2, 4.3, 9.4, 9.9, 8.6, 3.7, 9.8],
                       [2.1, 9.2, 6.3, 8.1537664, 6.5, 5.6, 8.7, 3.8],
                       [3.1, 7.2, 8.3, 9.4, 6.5, 7.55549692, 8.7, 8.8]])

    np.testing.assert_almost_equal(larvaemutated, goodsol)   

def test_cont_delta_larvaemutattion():
    """
    Test mutated larvae in a given position, cont mode
    mutation type: delta
    TARGET: larvaemutation -> cont_larvaemutation
    """ 
    
    larvae = np.array([[2.1, 4.2, 4.3, 9.4, 9.9, 8.6, 3.7, 9.8],
                       [2.1, 9.2, 6.3, 7.4, 6.5, 5.6, 8.7, 3.8],
                       [3.1, 7.2, 8.3, 9.4, 6.5, 7.6, 8.7, 8.8]])
    
    pos = np.array([[0, 3, 5]])
    mode = 'cont'
    mut_type = 'delta'
    grid = {'x': [2., 10.]}      # Discrete values between 2 and 10
    
    larvaemutation_function = get_larvaemutation_function(mode)
    larvaemutated = larvaemutation_function(larvae, pos, param_grid=grid, mut_type=mut_type, seed=13)
    
    goodsol = np.array([[4.1, 4.2, 4.3, 9.4, 9.9, 8.6, 3.7, 9.8],
                       [2.1, 9.2, 6.3, 7.4, 6.5, 5.6, 8.7, 3.8],
                       [3.1, 7.2, 8.3, 9.4, 6.5, 7.6, 8.7, 8.8]])

    np.testing.assert_almost_equal(larvaemutated, goodsol) 
    
def test_get_larvaemutation_function():
    """
    Test get_larvaemutation_function which returns mutation funcion for a mode
    TARGET: larvaemutation -> get_larvaemutation_function
    """
    
    f = get_larvaemutation_function('bin')
    assert 'function bin_larvaemutation' in str(f)
    f = get_larvaemutation_function('disc')
    assert 'function disc_larvaemutation' in str(f)

#!/usr/bin/env python
# coding=utf-8
###############################################################################
# run with:
# python -m pytest tests/test_larvaemutation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def test_cont_larvaemutattion():
    """
    Test mutated larvae in a given position, cont mode
    TARGET: larvaemutation -> cont_larvaemutation
    """   
    pass   
    

def test_get_larvaemutation_function():
    """
    Test get_larvaemutation_function which returns mutation funcion for a mode
    TARGET: larvaemutation -> get_larvaemutation_function
    """
    
    f = get_larvaemutation_function('bin')
    assert 'function bin_larvaemutation' in str(f)
    f = get_larvaemutation_function('disc')
    assert 'function disc_larvaemutation' in str(f)
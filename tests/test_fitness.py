#!/usr/bin/env python
# coding=utf-8
###############################################################################
# run with:
# python -m pytest tests/test_fitness.py
import numpy as np

from cro.fitness import max_ones

# ------------------------------------------------------
# reefinitialization module
# ------------------------------------------------------

def test_max_ones():
    """
    Test max_ones fitness function, it should return the percentage of 1's in the coral
    """  
    coral = np.array([[1,0,0,0,1],
                       [0,1,0,0,1], 
                       [0,1,1,0,1],
                       [0,1,1,1,1]])
    
    np.testing.assert_almost_equal(max_ones(coral), np.array([ 25.,  75.,  50.,  25., 100.]))
    coral = np.array([[2,0,0,0,9],
                       [0,3,0,0,1], 
                       [0,1,4,0,1],
                       [0,1,1,5,1]])
    
    np.testing.assert_almost_equal(max_ones(coral), np.array([  50.,  125.,  125.,  125.,  300.]))
    
def test_feature_selection():
    """
    To be added
    """
    pass

#!/usr/bin/env python
# coding=utf-8
###############################################################################
# run with:
# python -m pytest tests/test_larvaemutation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

#from cro.reefinitialization import bin_binary, disc_equal_range

# ------------------------------------------------------
# reefinitialization module
# ------------------------------------------------------

def test_bin_binary():
    """
    Test that corals in the population only contain values in {0, 1}
    """
    pass
    
    
def test_disc_equal_range():
    """
    Test that corals in population contain values specified in the grid
    """
    pass
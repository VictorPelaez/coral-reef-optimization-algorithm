#!/usr/bin/env python
# coding=utf-8
###############################################################################

"""
Module that contains all the functions that perform the larvae-mutation operators
Every function should start with the mode name it performs, followed by an underscore and the function name.
It should accept the following arguments:
    - brooders: individuals to be mutated
    - pos: selected positions to be mutated
    - delta: represents an increment or decrement in each mutation (it could be placed as arg in param_grid)
    - kwargs: extra arguments that the function might need (see
              https://stackoverflow.com/a/1769475 for an 
              explanation on kwargs)
It should return a mutated brooders
"""

import logging
import numpy as np

from .utils import get_module_functions

def bin_larvaemutation(brooders, pos, delta=None, **kwargs):
    """
    TB documented
    """
    (nbrooders, _) = brooders.shape
    brooders[range(nbrooders), pos] = np.logical_not(brooders[range(nbrooders), pos])
    return (brooders)

def disc_larvaemutation(brooders, pos, delta=1, **kwargs):
    """
    TB documented
    """
    try:
        param_grid = kwargs["param_grid"]
        seed = kwargs["seed"]
    except KeyError:
        raise ValueError("disc mode needs a param_grid as a dictionary")
    
    np.random.seed(seed)
            
    (nbrooders, lbrooders) = brooders.shape
    MM = np.zeros([nbrooders, lbrooders], int) # Mutation matrix

    for key, value in param_grid.items():
        m, M = value

    inc = (M - brooders[range(nbrooders), pos])
    dec = (brooders[range(nbrooders), pos] -m) 
    Inc = np.where(inc>dec)  
    Dec = np.where(inc<=dec) 
        
    MM[Inc[1], pos[Inc]] = np.random.randint(delta, np.min(inc[Inc])) if len(Inc[1]) !=0 else delta 
    MM[Dec[1], pos[Dec]] = -np.random.randint(delta, np.min(dec[Dec])) if len(Dec[1]) !=0 else -delta 

    return (brooders + MM)

"""""
UTILS
"""""
def get_larvaemutation_function(mode):
    """
    Returns the larvaemutation function for the given mode.
    If more than one function exists, return one randomly.
    """
    larvaemutation_functions = get_module_functions(__name__)
    mode_functions = [(name, func) for name, func in larvaemutation_functions.items()
                                   if name.startswith(mode)]
    if not mode_functions:
        raise ValueError("No initialization function for mode {}".format(mode))
    elif len(mode_functions) > 1:
        logging.warning("More than one initialization function for mode {}".format(mode))
        name, func = mode_functions[0]
        logging.info("Using {}".format(name))
    else:
        name, func = mode_functions[0]
        logging.info("Using {} for initializing the reef".format(name))

    return func
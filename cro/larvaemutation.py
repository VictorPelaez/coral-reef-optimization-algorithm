#!/usr/bin/env python
# coding=utf-8
###############################################################################

"""
Description:
    Module that contains all the functions that perform the larvae-mutation operators
    Every function should start with the mode name it performs, followed by an underscore and the function name
Input:
    - brooders: individuals to be mutated
    - pos: selected positions to be mutated
    - delta: represents an increment or decrement in each mutation (it could be placed as arg in param_grid)
    - kwargs: extra arguments that the function might need (see https://stackoverflow.com/a/1769475 for an explanation on kwargs)
Output:
    - brooders: mutated larvaes or brooders
"""

import logging
import numpy as np

from .utils import get_module_functions

def bin_larvaemutation(brooders, pos, delta=None, **kwargs):
    """
    Description:
        larvae-mutation in a binary mode   
    """
    (nbrooders, _) = brooders.shape
    brooders[range(nbrooders), pos] = np.logical_not(brooders[range(nbrooders), pos])
    return (brooders)

def disc_larvaemutation(brooders, pos, delta=1, **kwargs):
    """
    Description:
        larvae-mutation in a discrete mode   
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

def cont_larvaemutation(brooders, pos, delta=.1, **kwargs):
    """
    Description:
        larvae-mutation in a continuous mode
        mutation type:
            - "simple": simple gaussian mutation + larvae correction
            - "delta" : increase and decrease delta values controlling where (positions) increase or decrease

    """
    try:
        param_grid = kwargs["param_grid"]
        seed = kwargs["seed"]
        mut_type = kwargs["mut_type"]
    except KeyError:
        raise ValueError("continuous mode needs a param_grid as a dictionary")
        
    np.random.seed(seed)  
    (nbrooders, lbrooders) = brooders.shape
    MM = np.zeros([nbrooders, lbrooders], int) # Mutation matrix
   
    for key, value in param_grid.items():
        m, M = value
        
    if mut_type == 'simple':    
        brooders[range(nbrooders), pos] =  brooders[range(nbrooders), pos] + np.random.normal(0, 1, pos.shape)
        brooders = correction_larvaemutation(brooders, m, M)
    
    if mut_type == 'delta':
        inc = (M - brooders[range(nbrooders), pos])
        dec = (brooders[range(nbrooders), pos] -m) 
        Inc = np.where(inc>dec)  
        Dec = np.where(inc<=dec) 
        
        MM[Inc[1], pos[Inc]] = np.random.randint(delta, np.min(inc[Inc])) if len(Inc[1]) !=0 else delta 
        MM[Dec[1], pos[Dec]] = -np.random.randint(delta, np.min(dec[Dec])) if len(Dec[1]) !=0 else -delta 
        brooders = brooders + MM
        
    return (brooders)

def correction_larvaemutation(larvae, m, M):
    """
    Description:
        larvae correction after mutation operator  
    """          
    return np.interp(larvae, [m, M], [m, M])  

# ------------------------------------------------------
# UTILS
# ------------------------------------------------------

def get_larvaemutation_function(mode):
    """
    Description: 
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
        logging.info("Using {} for the brooding operator".format(name))

    return func

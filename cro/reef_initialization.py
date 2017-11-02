"""
Module that contains all the functions that perform the reef
initialization.

Every function should start with the mode name it performs,
followed by an underscore and the function name.

It should accept the following arguments:
    - M: Reef size
    - N: Reef size
    - r0: occupied/total ratio
    - L: coral length
    - kwargs: extra arguments that the function might need (see
              https://stackoverflow.com/a/1769475 for an 
              explanation on kwargs)

It should return a tuple with (REEF, REEFpob)
"""
import logging

import numpy as np

from utils import get_module_functions

def bin_binary(M, N, r0, L, **kwargs):
    """
    Each value in each coral in the reef is a boolean value, i.e,
    either a 0 or a 1
    """
    O = int(np.round(N*M*r0)) # number of occupied reefs 
    A = np.random.randint(2, size=[O, L])
    B = np.zeros([((N*M)-O), L], int)          
    REEFpob = np.concatenate([A, B]) # Population creation
    REEF = np.array((REEFpob.any(axis=1)),int) 
    return (REEF, REEFpob)

def disc_equalRange(M, N, r0, L, **kwargs):
    """
    Each value in each coral in the reef is an integer in the range
    specified by the keyword argument `param_grid`. `param_grid`
    must have the next format:

    >>> param_grid = {
            "x": [2, 10]   
        }

    where "x" can be basically anything, and its value is a list
    with both minimum and maximum value.
    In this example each coral will contain integers between 2 and 10
    """
    try:
        param_grid = kwargs["param_grid"]
    except KeyError:
        raise ValueError("disc mode needs a param_grid as a dictionary")

    O = int(np.round(N*M*r0)) # number of occupied reefs 
    for _, value in param_grid.items():
        valmax = (value[1] - value[0] + 1)
        A = np.random.randint(valmax, size=[O, L]) + value[0]
        B = np.zeros([((N*M)-O), L], int)
        REEFpob = np.concatenate([A,B]) # Population creation
        REEF = np.array((REEFpob.any(axis=1)),int)  
        return (REEF, REEFpob)

"""""
UTILS
"""""
def get_reefinit_function(mode):
    """
    Returns the init function for the given mode.
    If more than one function exists, return one randomly.
    """
    reef_init_functions = get_module_functions(__name__)
    mode_functions = [(name, func) for name, func in reef_init_functions.items()
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

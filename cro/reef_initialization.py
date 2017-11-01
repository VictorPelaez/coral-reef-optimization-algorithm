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

It should return a tuple with (REEF, REEFpob)
"""
import numpy as np

def bin_binaryReefInitialization(M, N, r0, L):
    """
    """
    O = int(np.round(N*M*r0)) # number of occupied reefs 
    A = np.random.randint(2, size=[O, L])
    B = np.zeros([((N*M)-O), L], int)          
    REEFpob = np.concatenate([A, B]) # Population creation
    REEF = np.array((REEFpob.any(axis=1)),int) 
    return (REEF, REEFpob)

def disc_equalRange(M, N, r0, L, param_grid):
    """
    """
    O = int(np.round(N*M*r0)) # number of occupied reefs 
    for key, value in param_grid.items():
        valmax = (value[1] - value[0] + 1)
        A = np.random.randint(valmax, size=[O, L]) + value[0]
        B = np.zeros([((N*M)-O), L], int)
        REEFpob = np.concatenate([A,B]) # Population creation
        REEF = np.array((REEFpob.any(axis=1)),int)  
        return (REEF, REEFpob)

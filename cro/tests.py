#!/usr/bin/env python
# coding=utf-8
###############################################################################

import os
import sys
import unittest
from cro import *

try:
    import numpy as np
except ImportError:
    import numpy 

class TestCRO(unittest.TestCase):

    def test_croCreation(self):
        L= 8
        N=2
        M=2
        fitness_coral = lambda coral: 1 # Dummy fitness
        cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
                  fitness_coral=fitness_coral, opt='max', L=L)
        (REEF, REEFpob) = cro.reefinitialization()
        self.assertEqual(REEFpob.shape, (N*M, L))
        
    def test_croInit(self):
        L= 8
        N=2
        M=2
        cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1,
                  fitness_coral=fitness_coral, opt='max', L=L)
        (REEF, REEFpob) = cro.reefinitialization()
        self.assertGreaterEqual(len(np.where(REEF!=0)[0]), 0)
        
            
            
###############################################################################

if __name__ == "__main__":
 
    unittest.main()
    sys.exit(0)


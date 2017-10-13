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

    
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)    

class TestCRO(unittest.TestCase):

    def test_croCreation(self):
        L= 8
        N=2
        M=2
        cro = CRO(Ngen=10, N=N, M=M, Fb=0.7, Fa=.1, Fd=.1, r0=.6, k=3, Pd=.1, opt='max', L=L)
        (REEF, REEFpob) = cro.reefinitialization()
        self.assertEqual(REEFpob.shape, (N*M, L))
            
            
###############################################################################

if __name__ == "__main__":
 
    unittest.main()
    sys.exit(0)


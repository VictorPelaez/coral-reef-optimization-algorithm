#!/usr/bin/env python
# coding=utf-8
###############################################################################

import os
import sys
import unittest
import awesome

try:
    import numpy as np
except ImportError:
    import numpy
class TestMethods(unittest.TestCase):
    def test_add(self):
        self.assertEqual(awesome.smile(), ":)")

###############################################################################

if __name__ == "__main__":
 
    unittest.main()
    sys.exit(0)


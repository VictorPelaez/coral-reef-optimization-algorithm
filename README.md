# coral-reef-optimization-algorithm
Coral Reefs Optimization (CRO) algorithm artificially simulates a coral reef, where different corals (which are the solutions for the considered optimization problem) grow and reproduce in a coral-reef, fighting with other corals for space

Flow diagram of the proposed CRO algorithm:

<img src = 'cro/assets/flow_diagram_cro.png' height ='500px'>


Install
-------

To install the library use pip:

    pip install cro


or clone the repo and just type the following on your shell:

    python setup.py install

Usage examples
--------------
Example of usage for max_ones problem: 

In this problem, [max_ones problem](https://github.com/Oddsor/EvolAlgo/wiki/Max-One-Problem), the health function is just the number of ones in the coral in percentage

The following results can be reproduced with command:  

```python
import numpy as np
import seaborn as sns #not necessary, just better plots
from cro.cro import *

## ------------------------------------------------------
## Parameters initialization

Ngen = 400                 # Number of generations
N  = 40                    # MxN: reef size
M  = 40                    # MxN: reef size
Fb = 0.8                   # Broadcast prob.
Fa = 0.1                   # Asexual reproduction prob.
Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
r0 = 0.7                   # Free/total initial proportion
k  = 3                     # Number of opportunities for a new coral to settle in the reef
Pd = 0.1                   # Depredation prob.
opt= 'max'                 # flag: 'max' for maximizing and 'min' for minimizing
L = 100

problem ='max_ones'
## ------------------------------------------------------

cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, opt, L,  problem_name=problem)
(REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness) = cro.fit()
```

<img src = 'cro/assets/max_ones_results/max_ones_ngen400_n40_m40_l100_fb08.png' height = '400px'> 

### Results for feature selection problem
(to be added)

## Folder structure
The following shows basic folder structure.
```
├── cro #package name
│   ├── cro.py # libs
│   ├── utils.py
│   ├── main_cro.py # used with args as shell script
├── assests
│   ├── data # dataset examples
│   |   ├── voice.csv
│   ├── max_ones_results

```

## Acknowledgements
This implementation has been based on Sancho Salcedo's idea and [this proyect](http://agamenon.tsc.uah.es/Personales/sancho/CRO.html) and tested with Python over ver3.0 on Windows 10

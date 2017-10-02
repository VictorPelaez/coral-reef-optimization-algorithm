#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import numpy as np
import argparse
from CRO import CRO

# python C:\Users\victor\Documents\Repositorios\coral_reef_optimization\main_cro.py --Ngen=400 --N=40 --L=100 --Fb=0.8 --problem=max_ones
# python C:\Users\victor\Documents\Repositorios\coral_reef_optimization\main_cro.py --Ngen=50 --N=10 --problem=feature_selection


def parse_args():
    desc = "CRO Algorithm"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--Ngen', type=int, default=20, help='The number of generation to run', required=True)
    parser.add_argument('--N', type=int, default=10, help='Reef size NxN', required=True)
    parser.add_argument('--L', type=int, default=100, help='Length bit string', required=False)
    
    parser.add_argument('--Fb', type=float, default=0.9, help='Broadcast prob', required=False)
    parser.add_argument('--Fa', type=float, default=0.1, help='Asexual reproduction prob', required=False)
    parser.add_argument('--Fd', type=float, default=0.1, help='Corals Fraction to be eliminated in depredation', required=False)
    parser.add_argument('--Pd', type=float, default=0.1, help='Depredation prob', required=False)  
    
    parser.add_argument('--opt_type', type=str, default='max', choices=['max', 'min'],
                        help='Optimization type: max for maximizing and min for minimizing', required=False)
    parser.add_argument('--problem', type=str, default='max_ones', choices=['max_ones', 'feature_selection'],
                        help='Problem to solve, ex. max_ones', required=True)
    parser.add_argument('--metric', type=str, default='r2', choices=['mse', 'mae', 'r2'],
                        help='Metric', required=False)
    parser.add_argument('--dataset', type=str, default='boston', help='Dataset', required=False)
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --epoch
    try:
        assert args.Ngen >= 1
    except:
        print('number of generation must be larger than or equal to one')
        
    return args
    

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    print(" [*] Testing CRO!")
    cro = CRO(Ngen=args.Ngen, N=args.N, Fb=args.Fb, Fa=args.Fa, Fd=args.Fd, r0=.7, k=3, Pd=args.Pd, opt=args.opt_type, L=args.L, seed=13,
              problem_name=args.problem, metric=args.metric, dataset_name=args.dataset)
    
    (REEF, REEFpob) = cro.reefinitialization ()
    REEFfitness = cro.fitness(REEFpob)
    Bestfitness = []
    Meanfitness = []   
    ke_allowed = 0.2 #percentage allowed with equal values in extreme depredation
    
    if args.opt_type=='max':
        print('Reef initialization:', np.max(REEFfitness))
        Bestfitness.append(np.max(REEFfitness))
    else: 
        print('Reef initialization:', np.min(REEFfitness))
        Bestfitness.append(np.min(REEFfitness))
    Meanfitness.append(np.mean(REEFfitness))
    
    
    for n in range(args.Ngen):
        ESlarvae = cro.broadcastspawning(REEF, REEFpob)
        ISlarvae = cro.brooding(REEF, REEFpob, 'op_mutation')
        
        # larvae fitness
        ESfitness = cro.fitness(ESlarvae)
        ISfitness = cro.fitness(ISlarvae)

        # Larvae setting
        larvae = np.concatenate([ESlarvae,ISlarvae ],axis=1)
        larvaefitness = np.concatenate([ESfitness, ISfitness])
        (REEF, REEFpob, REEFfitness) = cro.larvaesetting(REEF, REEFpob, REEFfitness, larvae, larvaefitness)
             
        # Asexual reproduction
        (Alarvae, Afitness) = cro.budding(REEF, REEFpob, REEFfitness)
        (REEF, REEFpob, REEFfitness) = cro.larvaesetting(REEF, REEFpob, REEFfitness, Alarvae, Afitness)
        
        if n!=args.Ngen:
            (REEF, REEFpob, REEFfitness) = cro.depredation(REEF, REEFpob, REEFfitness)    
            ke = int(np.round(ke_allowed*args.N*args.N))
            (REEF, REEFpob, REEFfitness) = cro.extremedepredation(REEF, REEFpob, REEFfitness, ke)
        
        if args.opt_type=='max': Bestfitness.append(np.max(REEFfitness))
        else: Bestfitness.append(np.min(REEFfitness))              
        Meanfitness.append(np.mean(REEFfitness))
        
        if (n%10==0) & (n!=args.Ngen):
            if args.opt_type=='max': print('Best-fitness:', np.max(REEFfitness), '\n', str(n/args.Ngen*100) + '% completado \n' );
            else: 
                print('Best-fitness:', np.min(REEFfitness), '\n', str(n/args.Ngen*100) + '% completado \n' );
    
    if args.opt_type=='max':
        print('Best-fitness:', np.max(REEFfitness), '\n', str(100) + '% completado \n' ) 
        ind_best = np.where(REEFfitness == np.max(REEFfitness))[0][0]
    else:
        print('Best-fitness:', np.min(REEFfitness), '\n', str(100) + '% completado \n' ) 
        ind_best = np.where(REEFfitness == np.min(REEFfitness))[0][0]
    
    print(REEFpob[:, ind_best])
    print(REEFfitness[ind_best])
    
    if args.problem=='feature_selection': print(cro.dataset_names(REEFpob[:, ind_best]))
    
    cro.plot_results( REEF, REEFpob, REEFfitness, Bestfitness, Meanfitness)
    # end
    print(" [*] Testing finished!")
   
    
if __name__ == '__main__':
    main()    
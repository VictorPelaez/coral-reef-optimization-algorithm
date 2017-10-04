#-*- coding: utf-8 -*-
from __future__ import division

import os
import time
import numpy as np

from sklearn import datasets, linear_model
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import metrics  

from utils import load_data 

class CRO(object):
    def __init__(self, Ngen, N, Fb, Fa, Fd, r0, k, Pd, opt, L=None, ke = 0.2,
                 seed=13, problem_name=None, metric=None, dataset_name=None, ml_problem=None, verbose=False):
        
        self.Ngen = Ngen
        self.N    = N
        self.Fb   = Fb
        self.Fa   = Fa
        self.Fd   = Fd              
        self.r0   = r0
        self.k    = k
        self.Pd   = Pd
        self.opt  = opt
        
        if (problem_name=='feature_selection') & (dataset_name=='boston'):
            self.L = datasets.load_boston().data.shape[1] 
            self.feature_names = datasets.load_boston().feature_names

        elif (problem_name=='feature_selection') & (dataset_name=='diabetes'):
            self.L = datasets.load_diabetes().data.shape[1]
            self.feature_names = ['age', 'sex', 'bmi', 'bp','s1', 's2', 's3', 's4', 's5', 's6']
            
        elif (problem_name=='feature_selection'): # other file name
            self.L = load_data(dataset_name).data.shape[1]
            self.feature_names = load_data(dataset_name).feature_names 
            
        else:
            self.L=L   
            
            
        self.ke = ke    
        self.seed = seed
        self.problem_name = problem_name
        self.metric = metric
        self.model_name = "CRO" 
        self.dataset = dataset_name
        self.ml_problem = ml_problem
        self.verbose = verbose
        
        if self.problem_name=='max_ones': self.metric='(%)'; self.opt='max'; self.dataset=''
        print("[*test] Initialization: ", self.problem_name, self.opt, self.metric, self.L, self.dataset)
    
    def reefinitialization (self):   
        """    
        function [REEF,REEFpob]=reefinitialization(M,N,r0,L)
        Description: Initialize the REEF with L-length random corals. r0 is the occupied/total rate.
        This function depends on the problem to be solved, max-ones in this case.   
        Input: 
            - MxN: reef size
            - r0: occupied/total rate
            - L: coral length
        Output:
            - REEF: reef matrix
            - REEFpob: population matrix
        """    
        np.random.seed(seed = self.seed)
        O = int(np.round(self.N*self.N*self.r0)) # number of occupied reefs    
        A = np.random.randint(2, size=[self.L, O])
        B = np.zeros([self.L, ( (self.N*self.N)-O)], int)
        REEFpob = np.concatenate([A,B], axis=1) # Population creation
        REEF = np.array((REEFpob.any(axis=0)),int) 
        return (REEF, REEFpob)
    
    def fitness(self, REEFpob):
        """
        Description: This function calculates the health function for each coral in the reef, 
        and is the most dependant function on the application.
        In this case (max-ones), the health function is just the number of ones in the coral in %
        """
        
        if self.problem_name=='max_ones': 
            return 100*(np.sum(REEFpob, axis=0)/REEFpob.shape[0])
        
        if self.problem_name=='feature_selection':
            np.random.seed(seed = self.seed)
            M = np.transpose(REEFpob)
            if (self.dataset=='boston'):
                df = datasets.load_boston()
                # ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
            elif (self.dataset=='diabetes'):
                df = datasets.load_diabetes()
                # ['age', 'sex', 'bmi', 'bp','s1', 's2', 's3', 's4', 's5', 's6']
            else: # other file name
                df = load_data(self.dataset)
                
        
            ftns = [] 
            for m in M:
                X, y = shuffle(df.data, df.target, random_state=self.seed)
                X = X.astype(np.float32)
                X = np.multiply(X, m)
                
                offset = int(X.shape[0] * 0.9)
                X_train, y_train = X[:offset], y[:offset]
                X_test, y_test = X[offset:], y[offset:]
               
                # #############################################################################
                if self.ml_problem =='regression':
                    # Fit regression model
                    params = {'n_estimators': 30, 'max_depth': 4, 'min_samples_split': 2}
                    #clf = linear_model.LinearRegression()
                    clf = ensemble.GradientBoostingRegressor(**params)                
                    #clf = ensemble.RandomForestRegressor(**params)
                
                if self.ml_problem =='classification':
                    # clf = ensemble.GradientBoostingClassifier()
                    clf = KNeighborsClassifier(2)
                
                clf.fit(X_train, y_train)   
                              
                # #############################################################################
                # Metrics
                if self.metric=='mse': # Mean squared error regression loss (best is 0)
                    fitness = metrics.mean_squared_error(y_test, clf.predict(X_test)) 
                elif self.metric=='r2': # R^2 (coefficient of determination) regression score function (best is 1.0)
                    fitness = metrics.r2_score(y_test, clf.predict(X_test))
                elif self.metric=='mae': # Mean absolute error regression loss
                    fitness = metrics.mean_absolute_error(y_test, clf.predict(X_test)) 
                elif self.metric=='auc':  
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))    
                    #fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
                    fitness = metrics.auc(fpr, tpr)
                else: 
                    print('[error] not a valid metric')
                    return -1
             
                ftns.append(fitness)
            return np.array(ftns)  

    def broadcastspawning(self, REEF,REEFpob): 
        """
        function [ESlarvae]=broadcastspawning(REEF,REEFpob,Fb,type)
        Create new larvae by external sexual reproduction. Cross-over operation

        Input: 
            - REEF: coral reef
            - REEFpob: reef population
            - Fb: fraction of broadcast spawners with respect to the overall amount of existing corals
            - mode: type of crossover depending on the type. type can be set to one of these options ('cont', 'disc','bin')
        Output:
            - ESlarvae: created larvae
        """
        Fb = self.Fb

        # get  number of spawners, forzed to be even (pairs of corals to create one larva)
        np.random.seed(seed = self.seed)
        nspawners = int(np.round(Fb*np.sum(REEF)))
        if (nspawners%2) !=0: 
            nspawners=nspawners-1

        # get spawners and divide them in two groups
        p = np.where(REEF!=0)[0]
        a = np.random.permutation(p)
        spawners = a[:nspawners]
        spawners1 = REEFpob[:, spawners[0:int(nspawners/2)]]
        spawners2 = REEFpob[:, spawners[int(nspawners/2):]]

        # get crossover mask for some of the methods below (one point crossover)
        (a,b) = spawners1.shape
        mask = np.random.randint(2, size=[a,b])
        mask = np.sort(mask, axis=0)
        notmask = np.logical_not(mask)

        ESlarvae1 = np.multiply(spawners1, np.logical_not(mask)) + np.multiply(spawners2, mask)
        ESlarvae2 = np.multiply(spawners2, np.logical_not(mask)) + np.multiply(spawners1, mask)
        ESlarvae = np.concatenate([ESlarvae1, ESlarvae2], axis=1)
        return ESlarvae

    def brooding(self, REEF, REEFpob, type_brooding='bin'):
        """
        function [ISlarvae]=brooding(REEF,REEFpob,Fb,type)
        Create new larvae by internal sexual reproduction.   
        Input:
            - REEF: coral reef, 
            - REEFpob: reef population, 
            - Fb: fraction of broadcast spawners with respect to the overall amount of existing corals 
            - type: type of crossover depending on the type. type can be set to one of these options ('cont', 'disc',   'bin')
        Output:
            - ISlarvae: created larvae
        """
        
        Fb = self.Fb
        
        #get the brooders
        np.random.seed(seed = self.seed)
        nbrooders= int(np.round((1-Fb)*np.sum((REEF))))
        p = np.where(REEF!=0)[0] 
        a = np.random.permutation(p)
        brooders= a[0:nbrooders]
        brooders=REEFpob[:, brooders]

        ISlarvae=np.zeros(brooders.shape)
        if type_brooding == 'bin':
            A = np.random.randint(2, size=brooders.shape)
            ISlarvae = (brooders + A) % 2
            
        if type_brooding == 'op_mutation':
            pos = np.random.randint(brooders.shape[0], size=(1, nbrooders))
            brooders[pos, range(brooders.shape[1])] = np.logical_not(brooders[pos, range(brooders.shape[1])])
            ISlarvae = brooders
        
        return ISlarvae

    def larvaesetting(self, REEF, REEFpob, REEFfitness, larvae, larvaefitness):
        """
        function [REEF,REEFpob]=larvaesetting(REEF,REEFpob,ESlarvae,ISlarvae)
        Settle the best larvae in the reef, eliminating those which are not good enough
        Input:    
            - REEF: coral reef
            - REEFpob: reef population
            - REEFfitness: reef fitness
            - larvae: larvae population
            - larvaefitness: larvae fitness
            - k0: number of oportunities for each larva to settle in the reef
            - opt: type of optimization ('min' or 'max')
        Output:
            - REEF: new coral reef
            - REEFpob: new reef population
            - REEFfitness: new reef fitness
        """
        k = self.k

        np.random.seed(seed = self.seed)
        Nlarvae = larvae.shape[1]-1
        a = np.random.permutation(Nlarvae)
        larvae = larvae[:, a]
        larvaefitness = larvaefitness[a]

        # Each larva is assigned a place in the reef to settle
        P = REEFpob.shape[1]
        nreef = np.random.permutation(P)
        nreef = nreef[0:Nlarvae]

        # larvae occupies empty places
        free = np.intersect1d(nreef, np.where(REEF==0))
        REEF[free]=1
        REEFpob[:, free] = larvae[:, :len(free)]
        REEFfitness[free] = larvaefitness[:len(free)]

        larvae = larvae[:, len(free):]  # update larvae
        larvaefitness = larvaefitness[len(free):] 

        # in the occupied places a fight for the space is produced
        nreef = np.random.permutation(P)
        ocup = np.intersect1d(nreef, np.where(REEF==1))
        Nlarvae = larvae.shape[1]
        
        for nlarvae in range(Nlarvae):
            for k0 in range(k):
                if(len(larvaefitness)==0) | (len(ocup)==0) : 
                    REEF = np.array((REEFpob.any(axis=0)),int)           
                    return (REEF, REEFpob, REEFfitness)
                
                ind = np.random.randint(len(ocup))
                
                #check if the larva is better than the installed coral
                if ( (self.opt=='max') & (larvaefitness[0] > REEFfitness[ocup[ind]])) | ( (self.opt=='min') 
                                                                      & (larvaefitness[0] < REEFfitness[ocup[ind]])): 
                    #settle the larva
                    REEF[ind] = 1
                    REEFpob[:, ind] = larvae[:, 0]
                    REEFfitness[ind] = larvaefitness[0]
                    
                    #eliminate the larva from the larvae list
                    larvae = larvae[:, 1:]  # update larvae
                    larvaefitness = larvaefitness[1:]
                    #eliminate the place from the occupied ones
                    ocup = np.delete(ocup, ind)   
        
        return (REEF,REEFpob,REEFfitness)


    def budding(self, REEF, REEFpob, fitness):
        """
        function  [Alarvae]=budding(REEF,pob,fitness,Fa)
        Duplicate the better corals in the reef.
        Input: 
            - REEF: coral reef 
            - pob: reef population
            - fitness: reef fitness 
            - Fa: fraction of corals to be duplicated
            - opt: type of optimization ('max' or 'min')
        Output: 
            - Alarvae: created larvae,
            - Afitness: larvae's fitness
        """
        Fa = self.Fa
        
        pob = REEFpob[:, np.where(REEF==1)[0]]
        fitness = fitness[np.where(REEF==1)]
        N = pob.shape[1]
        NA = int(np.round(Fa*N))
        
        if self.opt=='max': ind = np.argsort(-fitness); # max optimization problem!
        else: ind = np.argsort(fitness)    
            
        fitness = fitness[ind]
        Alarvae = pob[:,ind[0:NA]]
        Afitness = fitness[0:NA]
        return (Alarvae, Afitness)
    
    def depredation(self, REEF, REEFpob, REEFfitness):    
        """
        function [REEF,REEFpob,REEFfitness]=depredation(REEF,REEFpob,REEFfitness,Fd,Pd,opt)
        Depredation operator. A fraction Fd of the worse corals is eliminated with probability Pd
        Input:
            - REEF: coral reef
            - REEFpob: reef population
            - REEFfitness: reef fitness
            - Fd: fraction of the overall corals to be eliminated
            - Pd: probability of eliminating a coral
            - opt: type of optimization ('min' or 'max')
        Output:
            - REEF: new coral reef
            - REEFpob: new reef population
            - REEFfitness: new reef fitness
         """ 
        
        Fd = self.Fd
        Pd = self.Pd
        np.random.seed(seed = self.seed)
        
        if (self.opt=='max'):
            ind = np.argsort(REEFfitness)
        else: 
            ind = np.argsort(-REEFfitness)

        sortind = ind[:int(np.round(Fd*REEFpob.shape[1]))]
        p = np.random.rand(len(sortind))
        dep = np.where(p<Pd)[0]
        REEF[sortind[dep]] = 0
        REEFpob[:, sortind[dep]] = np.zeros([REEFpob.shape[0], len(dep)], int)
        
        # let's fix it: input the value for worst sol or empty sol
        if (self.opt=='min'): REEFfitness[sortind[dep]] = np.max(REEFfitness)
        else: REEFfitness[sortind[dep]] = np.min(REEFfitness)
        #REEFfitness[sortind[dep]] = 0
        
        return (REEF,REEFpob,REEFfitness)
    
    def extremedepredation(self, REEF, REEFpob, REEFfitness, ke):
        """    
        Allow only K equal corals in the reef, the rest are eliminated.
        Input:
            - REEF: coral reef
            - REEFpob: reef population
            - REEFfitness: reef fitness
            - ke: maximum number of allowed equal corals
        Output: 
            - REEF: new coral reef
            - REEFpob: new reef population
            - REEFfitness: new reef fitness
        """

        (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=1) 
        if len(np.where(np.sum(U, axis= 0)==0)[0]) !=0:
            zero_ind = int(np.where(np.sum(U, axis= 0)==0)[0])
            indices = np.delete(indices, zero_ind)
            count = np.delete(count, zero_ind)

        while np.where(count>ke)[0].size>0:
            higherk = np.where(count>ke)[0]
            REEF[indices[higherk]] = 0
            REEFpob[:, indices[higherk]] = np.zeros([REEFpob.shape[0], 1], int)
            
            # let's fix it: input the value for worst sol or empty sol
            if (self.opt=='min'): REEFfitness[indices[higherk]] = np.max(REEFfitness)
            else: REEFfitness[indices[higherk]] = np.min(REEFfitness)
            #REEFfitness[indices[higherk]] = 0

            (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=1) 
            if len(np.where(np.sum(U, axis= 0)==0)[0]) !=0:
                zero_ind = int(np.where(np.sum(U, axis= 0)==0)[0])
                indices = np.delete(indices, zero_ind)
                count   = np.delete(count, zero_ind)

        return (REEF,REEFpob,REEFfitness)
    
    
    def plot_results(self, REEF, REEFpob, REEFfitness, Bestfitness, Meanfitness):
            import matplotlib.pyplot as plt
            
            ngen = range(self.Ngen+1)  
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.plot(ngen, Bestfitness, 'b')     
            ax.plot(ngen, Meanfitness, 'r--')           
            plt.xlabel('Number of generation')
            plt.ylabel('Fitness function \n' + self.metric)
            
            if self.opt=='min': legend_place = (1,1);
            else: legend_place = (1,.3);
            plt.legend(['Best fitness', 'Mean fitness'], bbox_to_anchor=legend_place)
            
            
            titlepro = self.problem_name + ' Problem with Length vector (L): ' + str(self.L)
            titlepar = 'Ngen: '+ str(self.Ngen)+', N: '+str(self.N)+', Fb: '+str(self.Fb)+', Fa: '+str(self.Fa)+', Fd: '+str(self.Fd)+', Pd: '+ str(self.Pd)

            plt.title( titlepro+'\n'+ titlepar)
            
            ax.scatter(self.Ngen, Bestfitness[-1])
            ax.annotate('Best: ' + str(Bestfitness[-1]) , (self.Ngen, Bestfitness[-1]))
            
            plt.show()
            
    def dataset_names(self, REEFpob_best):
        names = np.array(self.feature_names)
        return names[REEFpob_best>0]
    
    def fit(self):
        
        Ngen = self.Ngen
        N = self.N
        verbose = self.verbose 
        opt = self.opt
        
        #Reef initialization
        (REEF, REEFpob) = self.reefinitialization ()
        REEFfitness = self.fitness(REEFpob)
        
        Bestfitness = []
        Meanfitness = []

        if opt=='max':
            if verbose: print('Reef initialization:', np.max(REEFfitness))
            Bestfitness.append(np.max(REEFfitness))
        else: 
            if verbose: print('Reef initialization:', np.min(REEFfitness))
            Bestfitness.append(np.min(REEFfitness))
        Meanfitness.append(np.mean(REEFfitness))


        for n in range(Ngen):
            ESlarvae = self.broadcastspawning(REEF, REEFpob)
            ISlarvae = self.brooding(REEF, REEFpob, 'op_mutation')

            # larvae fitness
            ESfitness = self.fitness(ESlarvae)
            ISfitness = self.fitness(ISlarvae)

            # Larvae setting
            larvae = np.concatenate([ESlarvae,ISlarvae],axis=1)
            larvaefitness = np.concatenate([ESfitness, ISfitness])
            (REEF, REEFpob, REEFfitness) = self.larvaesetting(REEF, REEFpob, REEFfitness, larvae, larvaefitness)

            # Asexual reproduction
            (Alarvae, Afitness) = self.budding(REEF, REEFpob, REEFfitness)
            (REEF, REEFpob, REEFfitness) = self.larvaesetting(REEF, REEFpob, REEFfitness, Alarvae, Afitness)

            if n!=Ngen:
                (REEF, REEFpob, REEFfitness) = self.depredation(REEF, REEFpob, REEFfitness)    
                (REEF, REEFpob, REEFfitness) = self.extremedepredation(REEF, REEFpob, REEFfitness, int(np.round(self.ke*N*N)) )

            if opt=='max': Bestfitness.append(np.max(REEFfitness))
            else: Bestfitness.append(np.min(REEFfitness))              
            Meanfitness.append(np.mean(REEFfitness))

            if (n%100==0) & (n!=Ngen):
                if (opt=='max') & (verbose): print('Best-fitness:', np.max(REEFfitness), '\n', str(n/Ngen*100) + '% completado \n' );
                if (opt=='min') & (verbose): print('Best-fitness:', np.min(REEFfitness), '\n', str(n/Ngen*100) + '% completado \n' );

        if opt=='max':
            if verbose: print('Best-fitness:', np.max(REEFfitness), '\n', str(100) + '% completado \n' ) 
            ind_best = np.where(REEFfitness == np.max(REEFfitness))[0][0]
        else:
            if verbose: print('Best-fitness:', np.min(REEFfitness), '\n', str(100) + '% completado \n' ) 
            ind_best = np.where(REEFfitness == np.min(REEFfitness))[0][0]


        if self.problem_name=='feature_selection': print(self.dataset_names(REEFpob[:, ind_best]))

        self.plot_results(REEF, REEFpob, REEFfitness, Bestfitness, Meanfitness)
        print('Best coral: ', REEFpob[:, ind_best])
        print('Best solution:', REEFfitness[ind_best])
        
        return (REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness)

   
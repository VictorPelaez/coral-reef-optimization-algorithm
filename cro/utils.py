import sys
from inspect import getmembers, isfunction

import pandas as pd
import numpy as np
from sklearn.utils import Bunch

# https://www.kaggle.com/adhok93/feature-importance-and-pca

def load_data(name):
    # csv file
    data = pd.read_csv('../cro/assets/data/' + name + '.csv')
    if name=='voice':
        data['label'] = data.label.apply(lambda x: 1 if x=='female' else 0) #one means female class
         
    feature_cols = [x for x in data.columns if (x!='label') & (x!='target')]
    
    # Add noisy features
    noisy_features = 10
    random_state = np.random.RandomState(0) 
    X = np.c_[np.array(data[feature_cols]), random_state.randn(data.shape[0], noisy_features)]
    feature_cols = feature_cols + noisy_features*['noise']
    
    dataset = Bunch(data= X, target=np.array(data.label), feature_names=feature_cols)
    return dataset

def get_module_functions(module_name):
    """
    Given the name of a module, return a dict with (name, function)
    for all the functions in the module
    """
    current_module = sys.modules[module_name]
    return dict(getmembers(current_module, predicate=isfunction))

import pandas as pd
import numpy as np
from sklearn import datasets

# https://www.kaggle.com/adhok93/feature-importance-and-pca
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

def load_data(name):
    # csv file
    data = pd.read_csv('assets/data/' + name + '.csv')
    if name=='voice':
        data['label'] = data.label.apply(lambda x: 1 if x=='female' else 0) #one means female class
         
    feature_cols = [x for x in data.columns if (x!='label') & (x!='target')]
    
    # Add noisy features
    noisy_features = 10
    random_state = np.random.RandomState(0) 
    X = np.c_[np.array(data[feature_cols]), random_state.randn(data.shape[0], noisy_features)]
    feature_cols = feature_cols + noisy_features*['noise']
    
    dataset = datasets.base.Bunch(data= X, target=np.array(data.label), feature_names=feature_cols)
    return dataset



"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import basicResults, makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


import warnings

#KNN
diamonds = pd.read_csv('./diamonds.csv')
diamondsX = diamonds.drop('is_over_$1000',1).copy().values
diamondsY = diamonds['is_over_$1000'].copy().values


diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])  

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  


params_diamonds= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")    
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,params_diamonds,'KNN','diamonds')        
    
    
    #madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
    #diamonds_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
    
    diamonds_final_params=diamonds_clf.best_params_
    
    pipeA.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeA,'KNN','diamonds')
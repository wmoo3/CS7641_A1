"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import warnings

#ANN
diamonds = pd.read_csv('./diamonds.csv')
diamondsX = diamonds.drop('is_over_$1000',1).copy().values
diamondsY = diamonds['is_over_$1000'].copy().values

diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)


pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = diamondsX.shape[1]
hiddens_diamonds = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
#alphasM = [10**-x for x in np.arange(-1,9.01,1/2)]

#d = d//(2**4)
params_diamonds = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_diamonds}

#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")    
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,params_diamonds,'ANN','diamonds')        
    
    
    #madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
    #diamonds_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}
    
    
    diamonds_final_params =diamonds_clf.best_params_
    diamonds_OF_params =diamonds_final_params.copy()
    diamonds_OF_params['MLP__alpha'] = 0
    
    #raise
    
    #
    
    #pipeM.set_params(**{'MLP__early_stopping':False})                   
    
    pipeA.set_params(**diamonds_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    makeTimingCurve(diamondsX,diamondsY,pipeA,'ANN','diamonds')
    
    
    #pipeM.set_params(**{'MLP__early_stopping':False})               
    pipeA.set_params(**diamonds_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    iterationLC(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','diamonds')                
    
    #pipeM.set_params(**{'MLP__early_stopping':False})                  
    pipeA.set_params(**diamonds_OF_params)
    pipeA.set_params(**{'MLP__early_stopping':False})               
    iterationLC(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','diamonds')                


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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import warnings

#ANN
poker = pd.read_csv('./poker-hand.csv')  

onehot=OneHotEncoder(categories='auto')
onehot.fit(poker.iloc[:,:10])
enc=pd.DataFrame(data=onehot.transform(poker.iloc[:,:10]).toarray(),columns=onehot.get_feature_names())
poker.drop(poker.iloc[:,:10],axis=1,inplace=True)
poker=poker.join(enc)
 
pokerX = poker.drop('has_hand',1).copy().values
pokerY = poker['has_hand'].copy().values

#pokerX, poker_tstX, pokerY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.9, random_state=0,stratify=pokerY)
poker_trgX, poker_tstX, poker_trgY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)



pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = pokerX.shape[1]
hiddens_poker = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
#alphasM = [10**-x for x in np.arange(-1,9.01,1/2)]

#d = d//(2**4)
params_poker = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_poker}

#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")    
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'ANN','poker')        
    
    
    #madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
    #poker_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}
    
    
    poker_final_params =poker_clf.best_params_
    poker_OF_params =poker_final_params.copy()
    poker_OF_params['MLP__alpha'] = 0
    
    #raise
    
    #
    
    #pipeM.set_params(**{'MLP__early_stopping':False})                   
    
    pipeA.set_params(**poker_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    makeTimingCurve(pokerX,pokerY,pipeA,'ANN','poker')
    
    
    #pipeM.set_params(**{'MLP__early_stopping':False})               
    pipeA.set_params(**poker_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    iterationLC(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','poker')                
    
    #pipeM.set_params(**{'MLP__early_stopping':False})                  
    pipeA.set_params(**poker_OF_params)
    pipeA.set_params(**{'MLP__early_stopping':False})               
    iterationLC(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','poker')                




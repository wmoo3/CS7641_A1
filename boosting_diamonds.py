"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import  basicResults,makeTimingCurve,iterationLC, dtclf_pruned
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import warnings

#boosting
diamonds = pd.read_csv('./diamonds.csv')   
diamondsX = diamonds.drop('is_over_$1000',1).copy().values
diamondsY = diamonds['is_over_$1000'].copy().values


#diamondsX, diamonds_tstX, diamondsY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.9, random_state=0,stratify=diamondsY)
diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)

#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
alphas = [-0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]

              
diamonds_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
#           'Boost__base_estimator__alpha':alphas}
                                   
         
diamonds_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=diamonds_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)


pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',diamonds_booster)])

#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,paramsA,'Boost','diamonds')        
    
    #
    #
    #madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
    #diamonds_final_params = {'n_estimators': 10, 'learning_rate': 1}
    #OF_params = {'learning_rate':1}
    
    diamonds_final_params = diamonds_clf.best_params_
    OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}
    
    ##
    pipeA.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeA,'Boost','diamonds')
    #
    pipeA.set_params(**diamonds_final_params)
    iterationLC(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','diamonds')                               
    pipeA.set_params(**OF_params)
    iterationLC(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','diamonds')                

             

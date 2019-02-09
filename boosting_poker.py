"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


import warnings


#boosting
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

#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
alphas = [-0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01]
alphas=[0.007]
   
poker_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[150],#[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
#           'Boost__base_estimator__alpha':alphas}
                                   
         
poker_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=poker_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)


pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',poker_booster)])

#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,paramsA,'Boost','poker')        
    
    #
    #
    #madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
    #poker_final_params = {'n_estimators': 10, 'learning_rate': 1}
    #OF_params = {'learning_rate':1}
    
    poker_final_params = poker_clf.best_params_
    OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}
    
    ##
    pipeA.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeA,'Boost','poker')
    #
    pipeA.set_params(**poker_final_params)
    iterationLC(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','poker')                               
    pipeA.set_params(**OF_params)
    iterationLC(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','poker')                

             

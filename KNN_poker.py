"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import basicResults, makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


import warnings

#KNN
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

metric=['minkowski','chebyshev']#['manhattan','euclidean','chebyshev']
p=[3.5]
K=[49]#np.arange(1,51,3)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])  

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  


params_poker= {'KNN__metric':metric,'KNN__n_neighbors':K,'KNN__weights':['distance'], 'KNN__p':p}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")    
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'KNN','poker')        
    
    
    #madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
    #poker_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
    
    poker_final_params=poker_clf.best_params_
    
    pipeA.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeA,'KNN','poker')


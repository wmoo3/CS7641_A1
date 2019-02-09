"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
from sklearn.svm import SVC
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from helpers import  basicResults,makeTimingCurve

import warnings

poker = pd.read_csv('./poker-hand.csv')  

onehot=OneHotEncoder(categories='auto')
onehot.fit(poker.iloc[:,:10])
enc=pd.DataFrame(data=onehot.transform(poker.iloc[:,:10]).toarray(),columns=onehot.get_feature_names())
poker.drop(poker.iloc[:,:10],axis=1,inplace=True)
poker=poker.join(enc)
 
pokerX = poker.drop('has_hand',1).copy().values
pokerY = poker['has_hand'].copy().values

#pokerX, poker_tstX, pokerY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)
poker_trgX, poker_tstX, poker_trgY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)        

#N_poker = poker_trgX.shape[0]

C = [0.1, 1, 5, 10]
coef0=[0, 1.0, 2.0]
degree=[2,3,4,5,6]
gamma = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0]


#RBF
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='rbf',random_state=55))])

#params_poker = {'SVM__C':C,'SVM__degree':degree, 'SVM__coef0':coef0}
params_poker = {'SVM__C':C, 'SVM__gamma':gamma}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'SVM_rbf','poker') 

    poker_final_params =poker_clf.best_params_
    
    pipeA.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeA,'SVM_rbf','poker')


#Poly
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='poly',random_state=55))])

params_poker = {'SVM__C':C,'SVM__degree':degree,'SVM__coef0':coef0}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'SVM_poly','poker') 

    poker_final_params =poker_clf.best_params_
    
    pipeA.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeA,'SVM_poly','poker')


 
'''  
#Linear
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='linear',random_state=55))])

params_poker = {'SVM__C':C}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    poker_clf = basicResults(pipeA,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'SVM_lin','poker') 

    poker_final_params =poker_clf.best_params_
    
    pipeA.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeA,'SVM_lin','poker')
'''
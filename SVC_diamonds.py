"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
from sklearn.svm import SVC
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  basicResults,makeTimingCurve

import warnings

diamonds = pd.read_csv('./diamonds.csv')  

diamondsX = diamonds.drop('is_over_$1000',1).copy().values
diamondsY = diamonds['is_over_$1000'].copy().values

#diamondsX, diamonds_tstX, diamondsY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)
diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)        

#N_diamonds = diamonds_trgX.shape[0]

C = [0.1, 1, 5, 10]
coef0=[0, 1.0, 2.0]
degree=[2,3,4,5,6]
gamma = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0]


#RBF
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='rbf',random_state=55))])

#params_diamonds = {'SVM__C':C,'SVM__degree':degree, 'SVM__coef0':coef0}
params_diamonds = {'SVM__C':C, 'SVM__gamma':gamma}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,params_diamonds,'SVM_rbf','diamonds') 

    diamonds_final_params =diamonds_clf.best_params_
    
    pipeA.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeA,'SVM_rbf','diamonds')



#Poly
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='poly',random_state=55))])

params_diamonds = {'SVM__C':C,'SVM__degree':degree,'SVM__coef0':coef0}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,params_diamonds,'SVM_poly','diamonds') 

    diamonds_final_params =diamonds_clf.best_params_
    
    pipeA.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeA,'SVM_poly','diamonds')


 
'''  
#Linear
pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SVC(kernel='linear',random_state=55))])

params_diamonds = {'SVM__C':C}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
                                                    
    diamonds_clf = basicResults(pipeA,diamonds_trgX,diamonds_trgY,diamonds_tstX,diamonds_tstY,params_diamonds,'SVM_lin','diamonds') 

    diamonds_final_params =diamonds_clf.best_params_
    
    pipeA.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeA,'SVM_lin','diamonds')
'''


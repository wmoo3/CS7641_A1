"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings





#Decision Tree with Pruning
def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    #fout=[]
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
        #features=clf.steps[-1][-1].features()
        #for f in features:
        #    if f>=0:
        #        fout.append(f)
        #print (np.unique(fout))
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return


diamonds = pd.read_csv('./diamonds.csv')   
diamondsX = diamonds.drop('is_over_$1000',1).copy().values
diamondsY = diamonds['is_over_$1000'].copy().values

diamondsX=np.delete(diamondsX,[1,2,3,4,5],1)

diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)

# Search for good alphas
alphas = [-0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
alphas=[0.0004]


pipeM = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])
    
#pipeM = Pipeline([('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['entropy'], 'DT__alpha':alphas,'DT__class_weight':['balanced']}
#params = {'DT__criterion':['entropy'], 'DT__alpha':alphas,'DT__class_weight':['balanced']}



with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    diamonds_clf = basicResults(pipeM, diamonds_trgX, diamonds_trgY, diamonds_tstX , diamonds_tstY, params,'DT','diamonds')

    diamonds_final_params = diamonds_clf.best_params_


    pipeM.set_params(**diamonds_final_params)
    makeTimingCurve(diamondsX,diamondsY,pipeM,'DT','diamonds')


    DTpruningVSnodes(pipeM,alphas,diamonds_trgX,diamonds_trgY,'diamonds')


























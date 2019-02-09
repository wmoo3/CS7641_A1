"""
Based on Jonathan Tay's CS7641 Assignment 1 Project

https://github.com/JonathanTay/CS-7641-assignment-1
"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned, makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

# Search for good alphas
alphas = [-0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
#alphas=[0.0006]

#max_depth=[10]

pipeM = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])
    
#pipeM = Pipeline([('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['gini','entropy'], 'DT__alpha':alphas,'DT__class_weight':['balanced']}#,'DT__max_depth':max_depth}
#params = {'DT__criterion':['gini'], 'DT__alpha':alphas,'DT__class_weight':['balanced']}



with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    poker_clf = basicResults(pipeM, poker_trgX, poker_trgY, poker_tstX , poker_tstY, params,'DT','poker')

    poker_final_params = poker_clf.best_params_


    pipeM.set_params(**poker_final_params)
    makeTimingCurve(pokerX,pokerY,pipeM,'DT','poker')


    DTpruningVSnodes(pipeM,alphas,poker_trgX,poker_trgY,'poker')


























# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 14:36:17 2018

@author: jwang
"""

import os
import gc
gc.collect()
print(os.listdir("input"))

import pandas as pd
import xgboost as xgb
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from xgboost.sklearn import XGBClassifier
from matplotlib.pylab import rcParams
from sklearn.grid_search import GridSearchCV   #Perforing grid search

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
y_train = train['renewal']
id_test = test['id']
del test['id']
del train['id']
#del train['renewal']

y_train.sum()


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1).drop(each,axis=1)
    return df

train=one_hot(train, ['sourcing_channel','residence_area_type'])
test=one_hot(test, ['sourcing_channel','residence_area_type'])

target = 'renewal'
predictors = [x for x in train.columns if x not in [target]]

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['renewal'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['renewal'], dtrain_predprob))
                    
#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#    feat_imp.plot(kind='bar', title='Feature Importances')
#    plt.ylabel('Feature Importance Score')
    
param_test0 = {'n_estimators':[40,60,80,100,120,140,160,180,200]}
gsearch0 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                                  n_estimators=param_test0,
                                                  max_depth=5,
                                                  min_child_weight=1,
                                                  gamma=0,
                                                  subsample=0.8,
                                                  colsample_bytree=0.8,
                                                  objective= 'reg:logistic',
                                                  nthread=4,
                                                  scale_pos_weight=1,
                                                  seed=27), 
param_grid = param_test0, scoring='roc_auc',iid=False, cv=5)
gsearch0.fit(train[predictors],train[target])
gsearch0.grid_scores_, gsearch0.best_params_, gsearch0.best_score_


param_test1 = {
 'max_depth':[3,5,7,9],
 'min_child_weight':[1,3,5]
}

gsearch1 = GridSearchCV(
        estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                  objective= 'reg:logistic', scale_pos_weight=1, seed=27), 
                                  param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)

gsearch1.fit(train[predictors],train[target])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[0,1,2]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=100, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

#param_test2b = {
# 'min_child_weight':[6,8,10,12]
#}
#gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
# min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test2b, scoring='roc_auc',iid=False, cv=5)
#gsearch2b.fit(train[predictors],train[target])
#
#modelfit(gsearch3.best_estimator_, train, predictors)
#gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

#xgb2 = XGBClassifier(
# learning_rate =0.1,
# n_estimators=1000,
# max_depth=4,
# min_child_weight=6,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
# nthread=4,
# scale_pos_weight=1,
# seed=27)
#modelfit(xgb2, train, predictors)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='roc_auc',iid=False, cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictors)

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=100,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)
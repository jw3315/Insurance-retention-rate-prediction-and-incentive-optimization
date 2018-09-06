# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:35:19 2018

@author: jwang
"""
import os
import gc
gc.collect()
print(os.listdir("input"))

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection._split import check_cv
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
y_train = train['renewal']
id_test = test['id']
del test['id']
del train['id']
del train['renewal']

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


## step1 renewal propensity
class XGBRegressorCV():
    
    def __init__(self, xgb_params=None, fit_params=None, cv=3):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv
    
    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        self.estimators_ = []
        
        for train, valid in cv.split(X, y):
            self.estimators_.append(
                xgb.XGBRegressor(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )

        return self
    
    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)

    
xgb_params = {
         'objective': 'reg:linear',
         'booster': 'gbtree',
         'learning_rate':0.01,
         'n_estimators':5000,
         'max_depth':3,
         'min_child_weight':1,
         'gamma':0,
#         'subsample':0.8,
#         'colsample_bytree':0.8,
         'alpha': 0.0,
#         'lambda': 0.0,
         'nthread':4,
         'scale_pos_weight':1,
         'seed':27

}

fit_params = {
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse',
        'verbose': False
    }

pipe = Pipeline(
    [
        ('xgb-cv', XGBRegressorCV(
                xgb_params=xgb_params,
                fit_params=fit_params,
                cv=10
            )
        )
    ]
)

X_train, y_train, X_test, id_test = train.values, y_train.values, test.values, id_test.values


#y_train_pred = pipe.predict(X_train)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

submission = pd.DataFrame()
submission['id'] = id_test
submission['renewal'] = y_pred

submission.loc[submission.renewal>1,'renewal'] =1  



### step2 optimize incentives
#Equation for the effort-incentives curve: Y = 10*(1-exp(-X/400))
#Equation for the % improvement in renewal prob vs effort curve: Y = 20*(1-exp(-X/5))

step2 =pd.DataFrame()
step2['id'] = id_test
step2['p_bench'] = y_pred
step2['premium'] = test.premium

from scipy.optimize import minimize
import math
def revenue(incentive):
    """revenue function"""
    effort = 10*(1-math.exp(-incentive/400))
    p_delta = 20*(1-math.exp(-effort/5))

    return -((p_bench + p_delta) * premium -incentive)

def constraint_1(x_t, s_0=1):
    """ Inventory constraint. s_0 - np.sum(x_t) >= 0.
    This is an inequality constraint. See more below.
    """
    effort = 10*(1-math.exp(-x_t/400))
    return s_0 -p_bench - 20*(1-math.exp(-effort/5))
    
    
def constraint_2(x_t):
    """ Positive demand. Another inequality constraint x_t >= 0 """
    return x_t

s_0=1
constraints = ({'type': 'ineq', 'fun':  lambda x, s_0=s_0:  constraint_1(x,s_0=s_0)},
                {'type': 'ineq', 'fun':  lambda x: constraint_2(x)})

result = [0]*len(id_test)
for i in range(len(id_test)):
    try:
        p_bench = step2.p_bench[i]
        premium = step2.premium[i]
        x0 = 0
        res = minimize(revenue, x0, method='SLSQP',constraints=constraints)
    except:
        res = 0
        
    result[i] = res.x[0]
    
submission['incentives'] = pd.Series(result,index=submission.index)
submission.loc[submission.incentives<0,'incentives'] =0
          
submission.to_csv('output/submission5.csv',
    index=None
)
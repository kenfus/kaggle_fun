#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:38:11 2019

@author: vtimmel
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import numpy as np

from data_cleaning import clean_fe

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_cleaned = clean_fe(train)

X_train, X_test, y_train, y_test = train_test_split(test_cleaned.loc[:,'Pclass':].values,
                                                    test_cleaned.loc[:,'Survived'].values,
                                                    test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_jobs = 1)

parameters = {'n_neighbors': np.arange(2,6)}

knc = GridSearchCV(knc, parameters, cv=5)

knc.fit(X_train, y_train)

knc_best = knc.best_estimator_

print('knc: ',knc.best_params_, knc.best_score_)


from sklearn.ensemble import RandomForestClassifier

#create a new random forest classifier
rf = RandomForestClassifier()

#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': np.arange(20, 30) }

#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data
rf_gs.fit(X_train, y_train)

#save best model
rf_best = rf_gs.best_estimator_

#check best n_estimators value
print('rfc: ',rf_gs.best_params_, rf_gs.best_score_)


from sklearn.svm import SVC

#create a new random forest classifier
svc = SVC(gamma = 'scale', probability=True)

#create a dictionary of all values we want to test for n_estimators
params_svc = {'C' : np.arange(1,4)}

#use gridsearch to test all values for n_estimators
svc_gs = GridSearchCV(svc, params_svc, cv=5)

#fit model to training data
svc_gs.fit(X_train, y_train)

#save best model
svc_best = svc_gs.best_estimator_

#check best n_estimators value
print('svc: ',svc_gs.best_params_, svc_gs.best_score_)


from sklearn.ensemble import GradientBoostingClassifier

#create a new random forest classifier
gbc = GradientBoostingClassifier()

#create a dictionary of all values we want to test for n_estimators
params_gbc = {'n_estimators': np.arange(25,45,1),'learning_rate': np.arange(0.01,0.4,0.01) }

#use gridsearch to test all values for n_estimators
gbc_gs = GridSearchCV(gbc, params_gbc, cv=5)

#fit model to training data
gbc_gs.fit(X_train, y_train)

#save best model
gbc_best = gbc_gs.best_estimator_

#check best n_estimators value
print('gbc: ',gbc_gs.best_params_, gbc_gs.best_score_)


import xgboost as xgb

params_xgb = {'max_depth': np.arange(3,7)}

xgb_model = xgb.XGBClassifier()

xgb_gs = GridSearchCV(xgb_model, params_xgb, cv=5)

xgb_gs.fit(X_train, y_train)

xgb_best = xgb_gs.best_estimator_

print('xgb: ',xgb_gs.best_params_, xgb_gs.best_score_)


from sklearn.ensemble import AdaBoostClassifier

#create a new random forest classifier
adc = AdaBoostClassifier()

#create a dictionary of all values we want to test for n_estimators
params_adc = {'n_estimators': np.arange(22,34,2), 'learning_rate': np.arange(0.01,1.0,0.051) }

#use gridsearch to test all values for n_estimators
adc_gs = GridSearchCV(adc, params_adc, cv=5)

#fit model to training data
adc_gs.fit(X_train, y_train)

#save best model
adc_best = adc_gs.best_estimator_

#check best n_estimators value
print('adc: ',adc_gs.best_params_, adc_gs.best_score_)


from sklearn.linear_model import LogisticRegression

#create a new logistic regression model
log_reg = LogisticRegression(solver = 'saga',tol = 1e-6, max_iter = 400, penalty = 'elasticnet', l1_ratio = 0.5)

#fit the model to the training data
log_reg.fit(X_train, y_train)

print('log_reg', log_reg.score(X_train,y_train))



from sklearn.ensemble import VotingClassifier

#create a dictionary of our models
estimators=[('knc',knc_best), ('rf', rf_best), ('log_reg', log_reg), 
            ('gbc', gbc_best), ('SVC',svc_best),('ADC',adc_best),('xgb',xgb_best)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')

#fit model to training data
ensemble.fit(X_train, y_train)

#test our model on the test data
print(ensemble.score(X_test, y_test))



#Create test data. Make a function next time, wtf.

test_cleaned = clean_fe(test, False)

submission = pd.DataFrame()
prediction = pd.DataFrame(ensemble.predict(test_cleaned.loc[:,'Pclass':].values))

submission['Survived']  = prediction[0]
submission['PassengerId']=test['PassengerId']
submission.to_csv('submission.csv',index = False)
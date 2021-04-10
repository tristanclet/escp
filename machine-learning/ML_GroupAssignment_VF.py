# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:36:32 2021

@author: josep
"""


    ## IMPORT TOOLS ##

import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import sklearn 
import xgboost


train = pd.read_csv(r'C:/Users/josep/Desktop/Machine Learning/Assignments/A4 & A5/train.csv')  
test = pd.read_csv(r'C:/Users/josep/Desktop/Machine Learning/Assignments/A4 & A5/test.csv')


    ## UNDERSTAND OUR DATA ##  
test.shape
test.head()
test.describe()


# No missing value 
train.isnull().sum()
test.isnull().sum()

# Visualize my target variable: label
sns.distplot(train['label'])

counter = Counter(train.label)
for k,v in counter.items():
    per = v / len(train.label) *100
    #print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per)) # var = 1 => 57 and var = -1 => 31371


    ## PREPARE MY DATA ## 
    
    
# Define X,y and X_test
y = train.label
X = train.drop(columns=['label'])

# Split training set for training and testing 
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.10, stratify=y)

      ## BUILD THE MODEL: XGB CLASSIFIER ## 
      
      
#Class weights       
from sklearn.utils import class_weight

class_weight = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)))


#Define model: XGBC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

model = XGBClassifier()
name = 'XGB'
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall')
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Make prediction on validation dataset
from sklearn.metrics import roc_curve, auc
model = XGBClassifier(obj='binary:logistic',seed=29)
model.fit(X_train, y_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, model.predict(X_train))
auc(false_positive_rate, true_positive_rate)

    
y_pred = model.predict(X_val)
print(model.score(X_val, y_val))

df = pd.DataFrame({'Actual' : y_val, 'Predicted' : y_pred})
print(df)

X_test = test.drop(columns=['label'])


    ## PREDICTION ## 
     
probability = model.predict_proba(X_test)[:,1]
print(probability)

final_result = pd.DataFrame({'id': test.id, 'Label' : probability})
final_result

final_result.describe()

final_result.loc[final_result['Label'] >= 0.0005625, 'Label'] = 1
final_result.loc[final_result['Label'] < 0.0005625, 'Label'] = -1

final_result.describe()
final_result.head(5)

final_result.Label.sum()

y_test = test[['id','label']]
y_test.label.sum()

from sklearn.metrics import accuracy_score
accuracy_score(y_test.label, final_result.Label)
 
 
     ## TO CSV
     
final_result['Probability'] = probability

final_result.to_csv('C:/Users/josep/Desktop/Machine Learning/Assignments/A4 & A5/ML_GA.csv', index = False)


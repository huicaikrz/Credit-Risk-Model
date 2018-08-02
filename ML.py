# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:49:50 2018

@author: Hui Cai
"""

import pandas as pd
import numpy as np
from __future__ import division
import random
import matplotlib.pyplot as plt

path = 'C:/Users/lenovo/Desktop/FT'
data = pd.read_csv(path + '/train_test_data_with_many_indexes.csv')
predictors = ["3-M TREASURY RATE", "STOCK INDEX RETURN", "DTD_Level", "SIZE_Level",
              "CSTA_Level", "NITA_Level", "DTD_Trend", "SIZE_Trend", "NITA_Trend",
              "MKBK", "CSTA_Trend"]
attributes = ['FirmInd', 'TimeInd', 'FirstInd', 'ExitInd', 'status']
data = data[attributes+predictors]

cor = data.iloc[:,5:].corr()

#resample, get default data
default = data[((data['ExitInd'] - data['TimeInd']) == 1) & (data['status'] == 1)]
n = len(default);K = 1
nondefault = data[(data['status'] == 0)]
pos = list(nondefault.index)
random.shuffle(pos)

train = pd.concat([default,nondefault.loc[pos[0:K*n]]])

pos = list(train.index)
random.shuffle(pos)
train = train.loc[pos]

X_train = np.array(train.iloc[:,5:])
Y_train = np.array(train['status'])

n = len(X_train)
X_test,Y_test = X_train[0:int(0.3*n)],Y_train[0:int(0.3*n)]
X_train,Y_train = X_train[int(0.3*n):],Y_train[int(0.3*n):]

#logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

log_reg.score(X_train,Y_train)
log_reg.score(X_test,Y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train,log_reg.predict(X_train))

from sklearn.metrics import precision_score,recall_score
precision_score(Y_train, log_reg.predict(X_train)) 
recall_score(Y_train,log_reg.predict(X_train))

#SVM
from sklearn.svm import SVC
clf = SVC(kernel='poly', class_weight='balanced',degree =2)
clf.fit(X_train,Y_train)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)

confusion_matrix(Y_train,clf.predict(X_train))
precision_score(Y_train, clf.predict(X_train)) 
recall_score(Y_train,clf.predict(X_train))

#plot the aggregate number of defaults

#prediction from the model

def model_predict(model):
    nondefault = data[(data['status'] == 0)]
    pos = list(nondefault.index)
    random.shuffle(pos)
    test = pd.concat([default,nondefault.loc[pos[0:K*n]]])
    pos = list(test.index)
    random.shuffle(pos)
    test = test.loc[pos]

    test['predict'] = model.predict(np.array(test.iloc[:,5:]))
    test['date'] = [pd.to_datetime(str(1990+d//12)+str(d%12),format = '%Y%m') if d%12 != 0 \
                else pd.to_datetime(str(1990+d//12 - 1)+str(12),format = '%Y%m') for d in test['TimeInd']]
    pre = test.groupby(['date'])
    return pre['predict'].sum()

pre1 = model_predict(log_reg)
pre2 = model_predict(clf)

data['date'] = [pd.to_datetime(str(1990+d//12)+str(d%12),format = '%Y%m') if d%12 != 0 \
                else pd.to_datetime(str(1990+d//12 - 1)+str(12),format = '%Y%m') for d in data['TimeInd']]

data.index = data['date']
default = data[((data['ExitInd'] - data['TimeInd']) == 1) & (data['status'] == 1)]
de = default.groupby('date')
de = de.count()

fig = plt.figure(figsize = (15,6))
ax1 = fig.add_subplot(121)
ax1.plot(de.index,de.iloc[:,0],label = 'True Value')
ax1.plot(pre1.index,pre1,label = 'Prediction')
plt.title('Logistic One Month Prediction')
plt.legend(loc = 'upper left')
ax2 = fig.add_subplot(122)
ax2.plot(de.index,de.iloc[:,0],label = 'True Value')

ax2.plot(pre2.index,pre2,label = 'Prediction')
plt.title('SVM One Month Prediction')
plt.legend(loc = 'upper left')
plt.show()

#plot ROC curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc = 'upper left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

from sklearn.model_selection import cross_val_predict
y_scores1 = cross_val_predict(log_reg, X_train, Y_train, cv=3,method="decision_function")
y_scores2 = cross_val_predict(clf, X_train, Y_train, cv=3,method="decision_function")

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(Y_train, y_scores1)
fpr1, tpr1, thresholds1 = roc_curve(Y_train, y_scores1)

roc_auc_score(Y_train, y_scores2)
fpr2, tpr2, thresholds2 = roc_curve(Y_train, y_scores2)

plot_roc_curve(fpr1, tpr1,label = 'Logistic')
plot_roc_curve(fpr2, tpr2,label = 'SVM')
plt.show()





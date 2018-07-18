# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:49:50 2018

@author: Hui Cai
"""

#进行resample,采样,是的两类差不多

#logistic regression
#SVM 
import pandas as pd
import numpy as np

path = 'C:/Users/lenovo/Desktop/FT'

data = pd.read_csv(path + '/train_test_data_with_many_indexes.csv')
predictors = ["3-M TREASURY RATE", "STOCK INDEX RETURN", "DTD_Level", "SIZE_Level",
              "CSTA_Level", "NITA_Level", "DTD_Trend", "SIZE_Trend", "NITA_Trend",
              "MKBK", "CSTA_Trend"]
attributes = ['FirmInd', 'TimeInd', 'FirstInd', 'ExitInd', 'status']
data = data[attributes+predictors]

#default in one month
default = data[((data['ExitInd'] - data['TimeInd']) == 1) & (data['status'] == 1)]
n = len(default);K = 2

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
from sklearn.metrics import f1_score
f1_score(Y_train,log_reg.predict(X_train))

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(Y_train,log_reg.predict(X_train))

from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(log_reg, X_train, Y_train, cv=3,method="decision_function")

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_train, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train, y_scores)
#SVM
from sklearn.svm import SVC

#linear underfitting
#degree with 5 seems overfitting
clf = SVC(kernel='poly', class_weight='balanced',degree =4)
clf.fit(X_train,Y_train)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)
#increase gamma, overfitting
#increase C, overfitting
#rbf not work
clf = SVC(kernel='rbf', class_weight='balanced',C =0.5,gamma = 2)
clf.fit(X_train,Y_train)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)

#perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron()

clf.fit(X_train,Y_train)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)

#LDA,GDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X_train,Y_train)

clf.score(X_train,Y_train)
clf.score(X_test,Y_test)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train,Y_train)

clf.score(X_train,Y_train)
clf.score(X_test,Y_test)


from sklearn.neural_network import MLPClassifier
#predict_proba get the prediction of probability
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
              hidden_layer_sizes=(100, 10), random_state=1)
clf.fit(X_train,Y_train)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)

from pymongo import MongoClient
client = MongoClient('mongodb://huicai:Caihui0824!@huicai-shard-00-00-g4vys.mongodb.net:27017,huicai-shard-00-01-g4vys.mongodb.net:27017,huicai-shard-00-02-g4vys.mongodb.net:27017/test?ssl=true&replicaSet=huicai-shard-0&authSource=admin')

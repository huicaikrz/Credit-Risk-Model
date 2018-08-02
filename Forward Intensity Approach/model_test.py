# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:54:54 2018

@author: Hui Cai
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

t_bail = 224
global t_bail

#样本内检验,故数据来自于para estimation.py文件
#all_alpha,all_beta为para estimation的估计

#return the cumulative distribution function for the horizon 
def predict(data,horizon,alpha,beta,bail = False):
    '''
    data为feature,horizon为预测期长度
    预测函数返回给定预测期()下
    '''
    deltat = 1/12
    x = np.array(data)
    if bail == True:
        ifbail = x[:,1] >t_bail
        f = np.array([np.exp(alpha[k][0]*np.exp(-alpha[k][1]*(x[:,1]-t_bail))*ifbail + np.dot(x[:,5:],alpha[k][3:])+alpha[k][2]) for k in range(horizon)])
    else:
        f = np.array([np.exp(np.dot(x[:,5:],alpha[k][1:])+alpha[k][0]) for k in range(horizon)])
    g_f = np.array([np.exp(np.dot(x[:,5:],beta[k][1:])+beta[k][0]) for k in range(horizon)])
    g = f + g_f
    pdf = np.exp(-np.cumsum(g,axis = 0)*deltat)*(1-np.exp(-f*deltat))
    cdf = pdf.T
    cdf = cdf.cumsum(axis =1)
    return cdf

#all_alpha,all_beta is from the parameter estimation
horizon = len(all_alpha)
#hor_t is the prediction period
hor_t = [1,3,6,12]
#size是指在算accuracy profile时的分点数目
size = 1000

#store accuracy ratio,AUC and KS
AR = [];AUC=[];KS = []
#遍历需要检验效果的horizon
for jj in hor_t:
    #依赖于para estimation的重采样结果,第jj-1个位置代表了那个预测期的采样数据
    data = all_data[jj-1]
    #计算累计分布函数（累积违约概率）
    cdf = predict(data,horizon,all_alpha,all_beta)
    #获取所有合适的样本点位置
    bb = data['TimeInd'] <= max(data['ExitInd']) -jj
    #获取违约公司的样本点
    aa = bb & (data['status'] == 1) & (data['ExitInd']-data['TimeInd']<=jj)
    #将违约公司的点标记为1
    stat = np.zeros((len(data),horizon))
    stat[aa,jj-1] = 1
    
    #the list for how many companies to predict as defaults
    Nkk = np.linspace(0,size,size)*sum(bb)/size
    Nkk =np.array([int(k) for k in Nkk])
    
    #calculate accuracy ratio
    a = pd.DataFrame(cdf[:,jj-1])
    a = a[bb]
    a = a.sort_values(by = 0,ascending = False)
    #calculate TP(True Positive)
    cap = [sum(stat[a.iloc[0:int(k)].index,jj-1]) for k in Nkk] #TP
    #number of defaults/ total companies 
    ThetaP = sum(stat[bb,jj-1])/sum(bb)
    #TPR
    capp = cap/sum(stat[bb,jj-1])
    #the area of the perfect model
    PerfAreaP = 1/2-ThetaP/2
    #approximate the area under the curve
    ar = np.sum((capp[1:] + capp[0:-1])*1/size/2,axis = 0)-0.5
    AR.append(ar/PerfAreaP)
    #calculate auc
    AUC.append(metrics.roc_auc_score(stat[bb,jj-1],cdf[bb,jj - 1]))
    #K-S, calculate FP/(FP+TN)
    FP = Nkk-np.array(cap)
    S = FP/(sum(bb)-sum(aa))
    plt.plot(capp)
    plt.plot(S)
    plt.show()
    #K-S
    K_S = capp - S
    #the position of the max K-S
    KS.append([max(K_S),np.where(K_S == max(K_S))])

print(AR)
print(AUC)
print(KS)
    
    
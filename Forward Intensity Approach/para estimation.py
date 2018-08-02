# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:31:12 2018
main function of the model
@author: Hui Cai
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random

#数据读取并选取需要的部分
path = 'C:/Users/lenovo/Desktop/FT'
data = pd.read_csv(path + '/train_test_data_with_many_indexes.csv')

predictors = ["3-M TREASURY RATE", "STOCK INDEX RETURN", "DTD_Level", "SIZE_Level",
              "CSTA_Level", "NITA_Level", "DTD_Trend", "SIZE_Trend", "NITA_Trend",
              "MKBK", "CSTA_Trend"]
attributes = ['FirmInd', 'TimeInd', 'FirstInd', 'ExitInd', 'status']
positives = ["DTD_Level", "DTD_Trend", "CSTA_Level", "CSTA_Trend", "NITA_Level", "NITA_Trend"]
frees = ["3-M TREASURY RATE", "STOCK INDEX RETURN", "SIZE_Level", "SIZE_Trend", "MKBK"]
data = data[attributes+frees+positives]
d = pd.DataFrame(data)

def check_status(t):
    '''
    get the status of the company before t+1
    return 4 array of True or False
    '''
    survive = (data['ExitInd'] > data['TimeInd'] + t + 1) | (data['ExitInd'] == data['TimeInd'] +t+1) &(data['status'] == 0)
    default = (data['ExitInd'] == data['TimeInd'] + t+ 1) & (data['status'] == 1)
    otherexit = (data['ExitInd'] == data['TimeInd']+t+1)&(data['status'] == 2)
    rest = (data['ExitInd'] < data['TimeInd']+t+1) | (data['FirstInd'] >data['TimeInd'])
    return survive,default,otherexit,rest

def loss_alpha(alpha,data,bail = False,t_bail = 224):
    '''
    The loss function of alpha, 
    with 'bail' to denote if including bail-out effect or not,
    t_bail is the starting point of the bail-out effect
    '''
    global survive,default,otherexit,rest
    x = np.array(data)
    if bail == True:
        ifbail = data['TimeInd'] > t_bail
        f = np.exp(alpha[0]*np.exp(-alpha[1]*(x[:,1]-t_bail))*ifbail+np.dot(x[:,5:],alpha[3:])+alpha[2])
    else:
        f = np.exp(np.dot(x[:,5:],alpha[1:])+alpha[0])
    lik_alpha = survive * np.exp(-f/12) + default * (1 - np.exp(-f/12)) + otherexit * np.exp(-f/12) + rest    
    return -np.sum(np.log(lik_alpha))

def loss_beta(beta,data):
    global survive,default,otherexit,rest
    x = np.array(data)
    g_f = np.exp(np.dot(x[:,5:],beta[1:])+beta[0])
    lik_beta = survive * np.exp(-g_f/12) + default + otherexit *(1 - np.exp(-g_f/12)) + rest    
    return -np.sum(np.log(lik_beta))

def Jacob_alpha(alpha,data,bail = False, t_bail = 224):
    '''
    calculate the gradient of loss function of alpha
    '''
    global survive,default,otherexit,rest
    x = np.array(data)
    #在feature前添加1列1,用于计算截距项的梯度   
    new_x = np.append(np.ones(len(x)).reshape(len(x),1),x[:,5:],axis = 1)
    #bailout下的梯度计算方法
    if bail == True:
        N = 14
        ifbail = data['TimeInd'] > t_bail
        #计算梯度的中间过程
        f = np.exp(alpha[0]*np.exp(-alpha[1]*(x[:,1]-t_bail))*ifbail+np.dot(x[:,5:],alpha[3:])+alpha[2])
        L = survive * np.exp(-f/12) + default * (1 - np.exp(-f/12)) + otherexit * np.exp(-f/12) + rest
        #calculate the gradient of the bail out parameters
        grad0 = np.exp(-f/12)/(-12)*f*np.exp(-alpha[1]*(x[:,1]-t_bail))*ifbail
        grad1 = grad0*alpha[0]*(t_bail-x[:,1])*ifbail
        J0 = -np.sum((survive*grad0 + default * (-grad0) +otherexit*grad0)/L)
        J1 = -np.sum((survive*grad1 + default * (-grad1) +otherexit*grad1)/L)
        X = survive * np.exp(-f/12)*f/(-12)+ default*(-np.exp(-f/12)*f/(-12))+ otherexit*np.exp(-f/12)*f/(-12)
        J = np.tile(X/L,(N-2,1)).T * new_x
        return np.array([J0,J1] + list(-np.sum(J,axis =0)))        
    #不带bailout时的梯度计算
    else:
        N = 12
        f = np.exp(np.dot(x[:,5:],alpha[1:])+alpha[0])
        X = survive * np.exp(-f/12)*f/(-12)+ default*(-np.exp(-f/12)*f/(-12))+ otherexit*np.exp(-f/12)*f/(-12)
        L = survive * np.exp(-f/12) + default * (1 - np.exp(-f/12)) + otherexit * np.exp(-f/12) + rest
        J = np.tile(X/L,(N,1)).T * new_x
        return -np.sum(J,axis =0)

def Jacob_beta(beta,data):
    global survive,default,otherexit,rest
    x = np.array(data);N = 12
    new_x = np.append(np.ones(len(x)).reshape(len(x),1),x[:,5:],axis = 1)
    g_f = np.exp(np.dot(x[:,5:],beta[1:])+beta[0])
    L = survive * np.exp(-g_f/12) + default + otherexit *(1 - np.exp(-g_f/12)) + rest    
    X = survive * np.exp(-g_f/12)*g_f/(-12) + otherexit*(-np.exp(-g_f/12)*g_f/(-12))
    J = np.tile(X/L,(N,1)).T * new_x
    return -np.sum(J,axis =0)

#由于未违约公司远多于违约公司,进行采样
def resample(d,t,K):
    '''
    该函数用于重采样,d是原始数据,t是预测期,K是采样后未违约点/违约点
    '''
    #取出样本内的公司
    bb = d['TimeInd'] <= max(d['ExitInd']) -t-1
    #取出违约公司的位置
    aa = bb & (d['status'] == 1) & ((d['ExitInd']-d['TimeInd'])<= (t+1))
    #取出所有没有exit的公司的位置
    cc = bb & (d['status'] == 0)
    #总共的违约公司样本点
    n = sum(aa)
    
    pos = list(d.index[cc])
    random.shuffle(pos)
    #form the new train data
    data = pd.concat([d[aa],d.loc[pos[0:K*n]]])
    data.index = list(range(len(data)))
    return data
    
#the following is the main function for parameter estimation
all_alpha = [];all_beta = []
horizon = 12
import time
begin = time.clock()
#估计时是否用bailout
bail = False

#某些参数具有经济意义,增加bound的限定,这里记录参数位置
bd_pos = [-1,-3,-4,-5]

["3-M TREASURY RATE", "STOCK INDEX RETURN", "DTD_Level", "SIZE_Level",
              "CSTA_Level", "NITA_Level", "DTD_Trend", "SIZE_Trend", "NITA_Trend",
              "MKBK", "CSTA_Trend"]
all_data = []

K = 3 
for t in range(horizon):
    #resample
    data = resample(d,t,K)
    #all_data是list,每一个位置对应不同horizon下的采样
    all_data.append(data)
    
    print(t)
    #for bailout, there will be 14 parameters
    if bail == True:
        N = 14
    else:
        N = 12
    bound = [(-inf,inf)]*N
    for i in bd_pos:
        bound[i] = (-inf,0)
    bound_b = bound[-12:]

    Alpha = np.array([0]*N)
    Beta = np.array([0]*12)
    survive,default,otherexit,rest = check_status(t)
    #参数估计,这里由于有bound限制,使用SLSQP方法
    opt_Alpha = minimize(loss_alpha,x0 = Alpha,method = 'SLSQP',args = (data,bail),bounds = bound,jac = Jacob_alpha)
    #if max(abs(opt_Alpha.jac)) > 1:
    #    opt_Alpha = minimize(loss_alpha,x0 = Alpha,method = 'TNC',bounds = bound,args = (data,bail))
    opt_Beta = minimize(loss_beta,x0 = Beta,args = (data,),method = 'SLSQP',bounds = bound_b,jac = Jacob_beta)
    #if max(abs(opt_Beta.jac)) > 1:
     #   opt_Beta = minimize(loss_alpha,x0 = Alpha,method = 'TNC',args = (data,bail),bounds = bound)
    
    #opt_Alpha = minimize(loss_alpha,x0 = Alpha,args = (data,),method = 'Newton-CG',jac = Jacob_alpha,hess = Hessian_alpha)
    #opt_Beta = minimize(loss_alpha,x0 = Alpha,args = (data,),method = 'Newton-CG',jac = Jacob_beta,hess = Hessian_beta)
    all_alpha.append(opt_Alpha.x)
    all_beta.append(opt_Beta.x)

print(time.clock()-begin)

all_data.to_csv


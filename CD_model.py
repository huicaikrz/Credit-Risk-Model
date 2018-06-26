# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:38:57 2018

@author: Hui Cai
"""

import numpy as np
import pickle

class CreditDefault:
    def __init__(self,alpha,beta,alpha_b):
        self.alpha = alpha
        self.beta = beta
        self.alpha_b = alpha_b
    
    def predict(self,data,bail):
        t_bail = 224;deltat = 1/12
        N = len(self.alpha)
        if bail == True:
            f = np.array([np.exp(self.alpha_b[k][0]*np.exp(-self.alpha_b[k][1]*(data[1]-t_bail)) + np.sum(data[5:]*self.alpha_b[k][3:])+self.alpha_b[k][2]) for k in range(N)])
        else:
            f = np.array([np.exp(np.sum(data[5:]*self.alpha[k][1:])+self.alpha[k][0]) for k in range(N)])
        
        g_f = np.array([np.exp(np.sum(data[5:]*self.beta[k][1:])+self.beta[k][0]) for k in range(N)])
        g = f + g_f
        pdf = np.exp(-np.cumsum(g,axis = 0)*deltat)*(1-np.exp(-f*deltat))     
        return pdf
        


 
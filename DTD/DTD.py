# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:41:43 2018

use DTD

@author: Hui Cai
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

#输入需要equity,

def L(mu,sigma):
    return -(n-1)/2*np.log(2*pi)-0.5*np.sum(np.log(sigma**2*1/4))\
            -np.sum(np.log(V(sigma)/A)) - np.sum(np.log(norm.cdf(d(sigma))))-\
            np.sum(np.log(V(sigma)*A))**2/2/sigma**2/0.25


def f(V,sigma,E,L):
    #V is the unknown variable
    d = (np.log(V/L) + (r + sigma**2/2)*1)/sigma

    return V*norm.cdf(d(sigma)) - np.exp(-r*1)*L*norm.cdf(d(sigma)-sigma*1) - E = 0
    




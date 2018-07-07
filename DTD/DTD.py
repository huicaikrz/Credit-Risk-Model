# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:41:43 2018

use DTD

@author: Hui Cai
"""
from math import pi
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

import pymysql

conn = pymysql.connect(host = '192.168.1.110',port = 33325,user = 'wind',
                       password = '123456',database = 'filesync')
cur = conn.cursor()
cur.execute('''
            SELECT WIND_CODE,ANN_DT,REPORT_PERIOD,TOT_ASSETS,TOT_SHRHLDR_EQY_INCL_MIN_INT
            FROM filesync.asharebalancesheet
            WHERE REPORT_PERIOD > 20041231 AND CRNCY_CODE = 'CNY' AND 
            STATEMENT_TYPE = 408001000 ORDER BY REPORT_PERIOD,WIND_CODE   
            ''')
bal = pd.DataFrame(list(cur.fetchall()),columns = ['WIND_CODE','ANN_DT',
                   'REPORT_PERIOD','TOT_CUR_ASSETS','TOT_CUR_LIAB','TOT_ASSETS',
                   'TOT_LIAB','TOT_SHRHLDR_EQY_INCL_MIN_INT'])



global r
#for each company, we have a list of BV of equity, 
#debt(short term debt + half of long-term debt), book value of asset

equity = []   
debt = []
BV_A = []
#do a loop to calculate V at different time
def L(mu,sigma,debt,equity,BV_A):
    n = len(debt)
    asset = [fsolve(f,[0],args = (sigma,equity[i],debt[i])).x for i in range(n)]
    all_d = [d(sigma,A) for A in asset]
    return -(n-1)/2*np.log(2*pi)-0.5*(n-1)*np.log(sigma**2*1/4)-\
            np.sum(np.log(asset[1:]/BV_A[1:])) - np.sum([np.log(norm.cdf(dt)) for dt in all_d])-\
            np.sum((np.log(asset[1:]*BV_A[0:-1]/asset[0:-1]/BV_A[1:])-(mu-sigma**2/2)
                    *0.25)**2/2/sigma**2/0.25)

d = lambda sigma,V: (np.log(V/L) + (r + sigma**2/2)*1)/sigma

def f(V,sigma,E,L):
    #V is the unknown variable
    return V*norm.cdf(d(sigma,V)) - np.exp(-r*1)*L*norm.cdf(d(sigma,V)-sigma*1) - E
    

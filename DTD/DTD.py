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
from scipy.optimize import minimize

import pymysql
conn = pymysql.connect(host = '192.168.1.110',port = 33325,user = 'wind',
                       password = '123456',database = 'filesync')
cur = conn.cursor()

cur.execute('''
            select wind_code,ann_dt,report_period,tot_assets,cap_stk,tot_cur_liab,tot_non_cur_liab
            from filesync.asharebalancesheet
            where report_period > 20041231 AND CRNCY_CODE = 'CNY' AND STATEMENT_TYPE = 408001000
            order by wind_code,report_period
            ''')
#balance sheet data
bal = pd.DataFrame(list(cur.fetchall()),columns = ['WIND_CODE','ANN_DT',
                   'REPORT_PERIOD','TOT_ASSETS','CAP_STK',
                   'ST_BORROW', 'LT_BORROW'])
cur.execute('''
            select s_info_windcode,trade_dt,s_dq_close from filesync.ashareeodprices 
            where trade_dt > 20041231
            order by s_info_windcode, trade_dt
            ''')
#get stock price
stk_price = pd.DataFrame(list(cur.fetchall()),columns = ['WIND_CODE','trade_dt','close'])
stk_price['trade_dt'] = [pd.to_datetime(str(d)) for d in stk_price['trade_dt']]

#data cleaning(drop na, drop code that start with A,coerce to be float)
bal = bal.dropna()
bal = bal[bal['WIND_CODE'] < '7'] 
bal['REPORT_PERIOD'] = [pd.to_datetime(str(date)) for date in bal['REPORT_PERIOD']]
bal[bal.columns[3:]] = bal[bal.columns[3:]].astype('float')
bal[bal.columns[3:]] = bal[bal.columns[3:]]/1000000

#get the intersection of the code of balance sheet and that of stock price
all_code = list(set(bal['WIND_CODE']).intersection(stk_price['WIND_CODE']))
all_code.sort()
stk_price.index = stk_price['WIND_CODE']
bal.index = bal['WIND_CODE']
bal = bal.loc[all_code]
stk_price = stk_price.loc[all_code]
bal['close'] = 0.

#match stocck price and balance sheet, put them into one dataframe 
for code in all_code:
    print(code)
    a = stk_price.loc[code]
    #if data length is not enough, drop
    if len(a) <= 100:
        continue
    try:
        if bal.loc[code,'close'].shape[0] <= 3.:
            continue
    except:
        if bal.loc[code,'close'] == 0:
            continue
    #stock price coerce to be monthly freq and get the last one
    a.index = a['trade_dt']
    a = a.resample('M').last()
    #reported but that does not have price at that month
    try:
        a = list(a.loc[bal.loc[code]['REPORT_PERIOD'],'close'])
    except:
        continue
    bal.loc[code,'close']= a

#drop the points that do not have price
bal = bal[bal['close'] != 0.]        
bal['equity'] = bal['CAP_STK']*bal['close']

#interest rate
global r
r = 0

#likelihood function
def L(para,debt,equity,BV_A,ht):
    mu,sigma = para[0],para[1]
    equity = np.array(equity); debt = np.array(debt)
    BV_A = np.array(BV_A)
    n = len(debt)
    asset = [fsolve(f,[equity[i]+debt[i]],args = (sigma,equity[i],debt[i]))[0] for i in range(n)]
    all_d = [d(sigma,asset[i],debt[i]) for i in range(n)]
    return (n-1)/2*np.log(2*pi)+0.5*np.sum(np.log(sigma**2*ht))+\
            np.sum(np.log(asset[1:]/BV_A[1:])) + np.sum([np.log(norm.cdf(dt)) for dt in all_d])+\
            np.sum((np.log(asset[1:]*BV_A[0:-1]/asset[0:-1]/BV_A[1:])-(mu-sigma**2/2)
                    *0.25)**2/2/sigma**2/ht)
#calculate d in the option pricing formula
d = lambda sigma,V,D: (np.log(V/D) + (r + sigma**2/2)*1)/sigma
#option pricing formula
f = lambda V,sigma,E,D: V*norm.cdf(d(sigma,V,D)) - np.exp(-r*1)*D*norm.cdf(d(sigma,V,D)-sigma*1) - E


all_code = list(set(bal['WIND_CODE']))
all_code.sort()
#a dataframe with all the parameters
all_para = pd.DataFrame([0]*len(all_code),index = all_code,columns = ['mu'])
all_para['sigma'] = 0.1

wrong = []
bal['DTD'] = float('nan')
#go through all code and do parameter estimation
for code in all_code:
    print(code)
    this_code = bal[bal['WIND_CODE'] == code]
    #the length of the code is too short
    if len(this_code) < 5:
        wrong.append(code)
        continue
    #get equity,debt and book value of asset
    equity = np.array(this_code['equity'])
    debt = np.array(this_code['ST_BORROW'] + 0.5*this_code['LT_BORROW'])
    BV_A = np.array(this_code['TOT_ASSETS'])
    
    #ht is time interval in year between two report period
    time = this_code['REPORT_PERIOD']
    ht = np.array([((time.iat[i+1] - time.iat[i])/30).days for i in range(len(time)-1)])
    ht = ht/12
    #para estimation
    mu,sigma = minimize(L,x0 = (0,0.2),method = 'Nelder-Mead',args = (debt,equity,BV_A,ht)).x
    all_para.loc[code,['mu','sigma']] = mu,sigma   
    
    #calculate asset and DTD
    asset = np.array([fsolve(f,[equity[i]+debt[i]],args = (sigma,equity[i],debt[i]))[0] for i in range(len(equity))])
    bal.loc[code,'DTD'] = (np.log(asset/debt) + (mu - sigma**2/2))/sigma






# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:08:16 2018

Altman Z-score model

@author: Hui Cai
"""
import pandas as pd
import numpy as np
import pymysql

conn = pymysql.connect(host = '192.168.1.110',port = 33325,user = 'wind',
                       password = '123456',database = 'filesync')
cur = conn.cursor()
cur.execute('''
            SELECT WIND_CODE,ANN_DT,REPORT_PERIOD,TOT_CUR_ASSETS,TOT_CUR_LIAB,
            TOT_ASSETS,TOT_LIAB,TOT_SHRHLDR_EQY_INCL_MIN_INT
            FROM filesync.asharebalancesheet
            WHERE REPORT_PERIOD > 19991231 AND CRNCY_CODE = 'CNY' AND 
            STATEMENT_TYPE = 408001000 ORDER BY REPORT_PERIOD,WIND_CODE   
            ''')
bal = pd.DataFrame(list(cur.fetchall()),columns = ['WIND_CODE','ANN_DT',
                   'REPORT_PERIOD','TOT_CUR_ASSETS','TOT_CUR_LIAB','TOT_ASSETS',
                   'TOT_LIAB','TOT_SHRHLDR_EQY_INCL_MIN_INT'])
cur.execute('''
            SELECT WIND_CODE,ANN_DT,REPORT_PERIOD,S_FA_EBIT,S_FA_RETAINEDEARNINGS
            FROM filesync.asharefinancialindicator
            WHERE REPORT_PERIOD > 19991231 AND CRNCY_CODE = 'CNY' 
            ORDER BY REPORT_PERIOD,WIND_CODE
            ''')
income = pd.DataFrame(list(cur.fetchall()),columns = ['WIND_CODE','ANN_DT',
                      'REPORT_PERIOD','S_FA_EBIT','S_FA_RETAINEDEARNINGS'])
bal = bal.set_index([bal['WIND_CODE'],bal['REPORT_PERIOD']])
income = income.set_index([income['WIND_CODE'],income['REPORT_PERIOD']])

#emerge Z = 3.25 + 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4
#Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 0.999X5
#把所有的都调出来，然后存储

#遍历每一天
#所有的股票算一个z-score

#join two dataframe accoeding to the index
data = pd.merge(bal,income)
data[data.columns[3:]] = data[data.columns[3:]].astype('float')
data = data.dropna()
#X1:(current assets - current liabilities)/total assets

x1 = (data['TOT_CUR_ASSETS'] - data['TOT_CUR_LIAB'])/data['TOT_ASSETS']
x2 = (data['S_FA_RETAINEDEARNINGS']/data['TOT_ASSETS'])
x3 = data['S_FA_EBIT']/data['TOT_ASSETS']
x4 = data['TOT_SHRHLDR_EQY_INCL_MIN_INT']/data['TOT_LIAB']

z_score = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4
z_score = pd.DataFrame(z_score,columns = ['z'])
z_score['WIND_CODE'] = data['WIND_CODE'];z_score['REPORT_PERIOD'] = data['REPORT_PERIOD']
z_score= z_score.set_index([data['WIND_CODE'],data['REPORT_PERIOD']])
z_score = z_score.dropna()

bankruptcy = z_score[z_score['z'] < 1.1].dropna()
#<1.1,1.1-2.6,>2.6

des = bankruptcy.groupby(by='WIND_CODE')
grey = z_score[(z_score < 2.6)&(z_score > 1.1)].dropna()
g = grey.groupby(by = 'WIND_CODE')


#自动update







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
#调取2000年之后的资产负债表数据
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
#调取2000年后的利润表数据
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

#emerging market formula Z = 3.25 + 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4

#合并资产负债表和利润表,数据缺失则直接dropna
data = pd.merge(bal,income)
#数据库调取decimal模式,转换为float
data[data.columns[3:]] = data[data.columns[3:]].astype('float')
data = data.dropna()
#计算四个indicator
x1 = (data['TOT_CUR_ASSETS'] - data['TOT_CUR_LIAB'])/data['TOT_ASSETS']
x2 = (data['S_FA_RETAINEDEARNINGS']/data['TOT_ASSETS'])
x3 = data['S_FA_EBIT']/data['TOT_ASSETS']
x4 = data['TOT_SHRHLDR_EQY_INCL_MIN_INT']/data['TOT_LIAB']

#z score的计算并标准化输出
z_score = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4
z_score = pd.DataFrame(z_score,columns = ['z'])
z_score['WIND_CODE'] = data['WIND_CODE'];z_score['REPORT_PERIOD'] = data['REPORT_PERIOD']
z_score= z_score.set_index([data['WIND_CODE'],data['REPORT_PERIOD']])
z_score = z_score.dropna()

bankruptcy = z_score[z_score['z'] < 1.1].dropna()
grey = z_score[(z_score['z'] < 2.6)&(z_score['z'] > 1.1)].dropna()
safe = z_score[z_score['z'] > 2.6]






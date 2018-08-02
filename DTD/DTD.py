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

conn = pymysql.connect(host='192.168.1.110', port=33325, user='wind',
                       password='123456', database='filesync')
cur = conn.cursor()
cur.execute(
    '''
    select trade_dt,b_info_rate from shiborprices
    where s_info_windcode = 'SHIBOR3M.IR'
    order by trade_dt
    '''
)
shibor = pd.DataFrame(list(cur.fetchall()), columns=['trade_dt', 'rate'])
shibor['trade_dt'] = [pd.to_datetime(d) for d in shibor['trade_dt']]
shibor.index = shibor['trade_dt']
shibor = shibor.resample('M').last()
shibor['rate'] = shibor['rate'].astype('float')
shibor['rate'] /= 100.

cur.execute('''
            select wind_code,ann_dt,report_period,tot_assets,cap_stk,st_borrow,lt_borrow
            from filesync.asharebalancesheet
            where report_period > 20061231 AND report_period < 20180101 AND 
            CRNCY_CODE = 'CNY' AND STATEMENT_TYPE = 408001000
            order by wind_code,report_period
            ''')

bal = pd.DataFrame(list(cur.fetchall()), columns=['WIND_CODE', 'ANN_DT',
                                                  'REPORT_PERIOD', 'TOT_ASSETS', 'CAP_STK',
                                                  'ST_BORROW', 'LT_BORROW'])

cur.execute('''
            select s_info_windcode,trade_dt,s_dq_close from filesync.ashareeodprices 
            where trade_dt > 20061231 AND trade_dt < 20180101
            order by s_info_windcode, trade_dt
            ''')

stk_price = pd.DataFrame(list(cur.fetchall()), columns=['WIND_CODE', 'trade_dt', 'close'])
stk_price['trade_dt'] = [pd.to_datetime(str(d)) for d in stk_price['trade_dt']]
stk_price['close'] = stk_price['close'].astype('float')
# data cleaning(drop na, drop code that start with A,coerce to be float)
bal = bal.dropna()
bal = bal[bal['WIND_CODE'] < '7']
bal['REPORT_PERIOD'] = [pd.to_datetime(str(date)) for date in bal['REPORT_PERIOD']]
bal[bal.columns[3:]] = bal[bal.columns[3:]].astype('float')
bal[bal.columns[3:]] = bal[bal.columns[3:]] / 1000000

# 取股价的code与资产股债表的code的交集
all_code = list(set(bal['WIND_CODE']).intersection(stk_price['WIND_CODE']))
all_code.sort()
stk_price.index = stk_price['WIND_CODE']
bal.index = bal['WIND_CODE']
bal = bal.loc[all_code]
stk_price = stk_price.loc[all_code]


# likelihood function
def L(para, debt, equity, BV_A, shibor, ht):
    mu, sigma = para[0], para[1]
    equity = np.array(equity);
    debt = np.array(debt)
    BV_A = np.array(BV_A)
    n = len(debt)
    asset = [fsolve(f, [equity[i] + debt[i]], args=(sigma, equity[i], debt[i], shibor[i]))[0] for i in range(n)]
    all_d = [d(sigma, asset[i], debt[i], shibor[i]) for i in range(n)]
    return (n - 1) / 2 * np.log(2 * pi) + 0.5 * np.sum(np.log(sigma ** 2 * ht)) + \
           np.sum(np.log(asset[1:] / BV_A[1:])) + np.sum([np.log(norm.cdf(dt)) for dt in all_d]) + \
           np.sum((np.log(asset[1:] * BV_A[0:-1] / asset[0:-1] / BV_A[1:]) - (mu - sigma ** 2 / 2)
                   * 0.25) ** 2 / 2 / sigma ** 2 / ht)


# calculate d in the option pricing formula
d = lambda sigma, V, D, r: (np.log(V / D) + (r + sigma ** 2 / 2) * 1) / sigma
# option pricing formula
f = lambda V, sigma, E, D, r: V * norm.cdf(d(sigma, V, D, r)) - np.exp(-r * 1) * D * norm.cdf(
    d(sigma, V, D, r) - sigma * 1) - E

all_code = list(set(bal['WIND_CODE']))
all_code.sort()
# 存放所有的参数
all_para = pd.DataFrame(float('nan'), index=all_code, columns=['mu', 'sigma', 'n_period'])
outcome = pd.DataFrame(columns=['date', 'code', 'DTD', 'asset'])
wrong = []
bal['close'] = float('nan')
# go through all code and do parameter estimation

for code in all_code[2459:]:
    print(code)
    this_code = bal[bal['WIND_CODE'] == code]
    this_code.index = this_code['REPORT_PERIOD']
    # resample the price to be monthly price
    p = stk_price.loc[code]
    if len(p) <= 20:
        continue
    p.index = [p['trade_dt']]
    p = p.resample('M').last()
    p = p[p.index >= this_code['REPORT_PERIOD'].iat[0]]
    p = p.dropna()

    pos = set(p.index).intersection(this_code['REPORT_PERIOD'])
    # 月度股价向report period 匹配，没有股价的日子是na
    this_code.loc[pos, 'close'] = p.loc[pos, 'close']

    # 有报告期但是无股价的则用最近的值填充,极端情况是一个价格都没有
    this_code = this_code.ffill().bfill()
    n_period = len(this_code)

    # 报告期向月度股价匹配
    for fin in ['CAP_STK', 'ST_BORROW', 'LT_BORROW', 'TOT_ASSETS']:
        p.loc[pos, fin] = this_code.loc[pos, fin]
    # 然后前向后向填充
    p = p.ffill().bfill()

    # the length of the code is too short
    if len(this_code) <= 2:
        wrong.append(code)
        continue

    # 获取balance sheet中的几个指标
    equity = np.array(this_code['close'] * this_code['CAP_STK'])
    debt = np.array(this_code['ST_BORROW'] + 0.5 * this_code['LT_BORROW'])
    BV_A = np.array(this_code['TOT_ASSETS'])
    shi = np.array(shibor.loc[this_code.index, 'rate'])
    # 计算两个report period之间的时间间隔,年化
    time = this_code['REPORT_PERIOD']
    ht = np.array([((time.iat[i + 1] - time.iat[i]) / 30).days for i in range(len(time) - 1)])
    ht = ht / 12.

    # 参数估计
    model = minimize(L, x0=(0, 0.2), method='Nelder-Mead', args=(debt, equity, BV_A, shi, ht))
    if model.success == True:
        mu, sigma = model.x
        all_para.loc[code, ['mu', 'sigma', 'n_period']] = mu, sigma, n_period
    else:
        continue
        # 利用月度数据计算每个有股价DTD
    equity = np.array(p['close'] * p['CAP_STK'])
    debt = np.array(p['ST_BORROW'] + 0.5 * p['LT_BORROW'])
    BV_A = np.array(p['TOT_ASSETS'])
    shi = np.array(shibor.loc[p.index, 'rate'])
    # 计算每个月的asset
    asset = np.array(
        [fsolve(f, [equity[i] + debt[i]], args=(sigma, equity[i], debt[i], shi[i]))[0] for i in range(len(equity))])
    n = len(asset)
    # 格式化成dataframe输出

    out = pd.DataFrame([p.index, [code] * n, (np.log(asset / debt) + (mu - sigma ** 2 / 2)) / sigma, asset]).T
    out.columns = ['date', 'code', 'DTD', 'asset']
    outcome = outcome.append(out)

outcome.to_csv('F:/HuiCai/outcome.txt')
all_para.to_csv('F:/HuiCai/all_para.txt')

conn = pymysql.connect(host='192.168.1.110', port=33330, user='root',password= 'alphai_mysql_passwd',
                        database='financeDB')
cur = conn.cursor()
for i in range(len(outcome)):
    print(i)
    d = outcome.iloc[i].date
    m = d.month;y = d.year
    if m < 10:
        m = '0'+str(m)
    else:
        m = str(m)
    y = str(y)
    da = y+m
    code = outcome.iloc[i].code
    DTD = outcome.iloc[i].DTD
    id = code + '&&&' + da + '&&&' + 'dtd'
    if DTD == float('inf'):
        cur.execute('''INSERT INTO feature_tmp (id,stock_id, month, name,value) VALUES (%s, %s, %s, %s, %s)''',
                    (id,code,da,'dtd','inf'))
        continue
    cur.execute('''INSERT INTO feature_tmp (`id`, `stock_id`, `month`, `name`, `value`) VALUES (%s, %s, %s, %s, %s)''', (id,code,da,'dtd',str(DTD)))

conn.commit()
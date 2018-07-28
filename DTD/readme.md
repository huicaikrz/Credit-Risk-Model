#DTD

## data source

| database | table | item | decription |
|:------------- |:---------------:| :-------------:| :-------------:|
| filesync      |shiborprices     | SHIBOR3M.IR | 3 month shibor rate
| filesync      |asharebalancesheet |wind_code,ann_dt,report_period,tot_assets,cap_stk,st_borrow,lt_borrow | equity code, announce date, report period, total assets, capital stock, short term borrow, long term borrow
| filesync      |ashareeodprices    |s_info_windcode,trade_dt,s_dq_close | equity code, trade date, close price

## feature engineering
drop nan

equity = s_dq_close\*cap_stk </br>
debt = st_borrow + 0.5\*lt_borrow
book value of asset = total asset

## construct model

view equity as a call option and E = max(0,A-D) </br>
where E is equity, A is fair value of asset and D is debt </br>
based on BS model, we could derive the relationship between E, A and D </br>
E can be estimated through the stock price in the market, D is the short-term debt + half of the long-term debt </br>

Based on the likelihood function, we could estimate the parameters \mu, \sgima, calculate A by solving function and then calculate DTD. 

## predict

DTD can be viewed as the probability of the default probability to some sense. We select 5% of the points and view them as problematic points.




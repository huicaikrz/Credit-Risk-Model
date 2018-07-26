# Altman zscore model
https://en.wikipedia.org/wiki/Altman_Z-score
## data source

| database | table | item | decription |
|:------------- |:---------------:| :-------------:| :-------------:|
| filesync      |shiborprices     |SHIBOR3M.IR | 3 month shibor rate
| filesync      |asharebalancesheet |wind_code,ann_dt,report_period| equity code, announce date, report period
| filesync      |asharebalancesheet |tot_assets, cap_stk,st_borrow,lt_borrow | total assets, capital stock, short term borrow, long term borrow
| filesync      |ashareeodprices    |s_info_windcode,trade_dt,s_dq_close | equity code, trade date, daily close price

## feature engineering
drop nan
## construct model
zscore
## predict
if zscore > threshold, then blablabla ....

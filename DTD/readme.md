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

##construct model
$\sum_{i=0}^N




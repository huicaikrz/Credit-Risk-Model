# DTD

## data source

| database | table | item | decription |
|:------------- |:---------------:| :-------------:| :-------------:|
| filesync      |shiborprices     | SHIBOR3M.IR | 3 month shibor rate
| filesync      |asharebalancesheet |wind_code,ann_dt,report_period,tot_assets,cap_stk,st_borrow,lt_borrow | equity code, announce date, report period, total assets, capital stock, short term borrow, long term borrow
| filesync      |ashareeodprices    |s_info_windcode,trade_dt,s_dq_close | equity code, trade date, close price

## feature engineering
1. drop nan

2. feature construction
    * equity = s_dq_close\*cap_stk </br>
    * debt = st_borrow + 0.5\*lt_borrow
    * book value of asset = total asset

## construct model

view equity as a call option and E = max(0,A-D),where E is equity, A is fair value of asset and D is debt

Based on BS model, we could derive the relationship between E, A and D

$`
{\displaystyle
\begin{aligned}
&E_t = V_tN(d_t) - exp(-r(T-t))LN(d_t-\sigma\sqrt{T-t}) \\
&d_t = \frac{\log(V_t/L)+(r+\sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}
\end{aligned}
}
`$

E can be estimated through the stock price in the market, D is the short-term debt + half of the long-term debt

Based on the likelihood function below, we could estimate the parameters $`\mu`$ and $`\sigma`$, 

$`
{\displaystyle
\begin{aligned}
L(\mu,\sigma,\delta) = &-\frac{n-1}{2}\log(2\pi)-\frac{1}{2}\sum_{t=2}^{n}\log(\sigma^2h_t) \\
&- \sum_{t=2}^{n}\log(\hat{V}_t(\sigma,\delta)/A_t)-\sum_{t=2}^{n}\log(N(\hat{d}_t(\sigma,\delta)))-\sum_{t=2}^{n}\frac{1}{2\sigma^2h_t}[\log(\frac{\hat{V}_t(\sigma,\delta)}{\hat{V}_{t-1}(\sigma,\delta)}\times\frac{A_{t-1}}{A_t})-(\mu-\sigma^2/2)h_t]^2
\end{aligned}
}
`$

then calculate $A_t$ by solving the equation of $`E_t`$ and $`d_t`$ and we can calculate DTD with the formula below

$`
{\displaystyle
DTD_t = \frac{\log(V_t/L)+(\mu-\sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}
}
`$

## predict

DTD can be viewed as the probability of the default probability to some sense. We select 5% of the points and view them as problematic points.




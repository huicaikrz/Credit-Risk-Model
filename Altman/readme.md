# Altman zscore model
https://en.wikipedia.org/wiki/Altman_Z-score
## data source

| database | table | item | decription |
|:------------- |:---------------:| :-------------:| :-------------:|
| filesync      |asharebalancesheet |WIND_CODE,ANN_DT,REPORT_PERIOD,TOT_CUR_ASSETS,TOT_CUR_LIAB,TOT_ASSETS,TOT_LIAB,TOT_SHRHLDR_EQY_INCL_MIN_INT| equity code, announce date, report period
| filesync      |ashareeodprices    |WIND_CODE,ANN_DT,REPORT_PERIOD,S_FA_EBIT,S_FA_RETAINEDEARNINGS | 

## feature engineering
drop nan

There are four features which are calculated from the data above

x1 = (current assets − current liabilities) / total assets </br>
x2 = retained earnings / total assets </br>
x3 = earnings before interest and taxes / total assets </br>
x4 = book value of equity / total liabilities </br>

## construct model
zscore = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

## predict
Zones of discriminations:</br>
Z > 2.6 – “Safe” Zone</br>
1.1 < Z < 2.6 – “Grey” Zone</br>
Z < 1.1 – “Distress” Zone</br>

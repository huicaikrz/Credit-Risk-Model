# NUS

## data source

parameter estimation: csv file
model test: 'all_data','all_alpha','all_beta' comes from the outcome of parameter estimation

## feature engineering
features: ["3-M TREASURY RATE", "STOCK INDEX RETURN", "DTD_Level", "SIZE_Level",
              "CSTA_Level", "NITA_Level", "DTD_Trend", "SIZE_Trend", "NITA_Trend",
              "MKBK", "CSTA_Trend"]

the parameters of CSTA_Trend, NITA_Trend, SIZE_Trend, DTD_Trend should be less than zero

## construct model

maximize the likelihood function through python package scipy <\br>

Since some of the parameters need to be smaller than zero, here we use the SLSQP method, which could add bounded condition </br>


## predict

model test: here we only do in-sample model test including calculating accuracy ratio, AUC, K-S

## 







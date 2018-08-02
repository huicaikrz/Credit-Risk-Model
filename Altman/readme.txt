Altman z-score模型

https://en.wikipedia.org/wiki/Altman_Z-score

Z-Score bankruptcy model (emerging markets):

Z = 3.25 + 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4

where:
X1 = (current assets − current liabilities) / total assets
X2 = retained earnings / total assets
X3 = earnings before interest and taxes / total assets
X4 = book value of equity / total liabilities

Zones of discriminations:
Z > 2.6 – “Safe” Zone
1.1 < Z < 2.6 – “Grey” Zone
Z < 1.1 – “Distress” Zone

Altman-Zscore.py用于回测模型优劣, 
从wind数据库中调取2000年之后所有上市公司的资产负债表和利润表数据
缺失值直接drop




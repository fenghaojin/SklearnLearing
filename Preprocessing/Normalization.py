from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

data=[[-1,2],[-0.5,6],[0,10],[1,18]]
#data=pd.DataFrame(data)

#归一化
scaler=MinMaxScaler()
scaler=scaler.fit(data)#数据特征过多时用partial_fit
result=scaler.transform(data)

print(result)

#结果逆转
print(scaler.inverse_transform(result))

scaler_1=MinMaxScaler(feature_range=(5,10))
result_1=scaler_1.fit_transform(data)
print(result_1)

#numpy实现归一化
data=np.array(data)
result_2=(data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
print(result_2)

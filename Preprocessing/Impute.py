from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

df=pd.read_csv("./datasets/Narrativedata.csv",index_col=0)
print(df.info())
Age=df.loc[:,"Age"].values.reshape((-1,1))

imp_mean=SimpleImputer()
imp_median=SimpleImputer(strategy="median")
imp_0=SimpleImputer(strategy="constant",fill_value=0)

result_1=imp_mean.fit_transform(Age)
result_2=imp_median.fit_transform(Age)
result_3=imp_0.fit_transform(Age)

Embarked=df.loc[:,"Embarked"].values.reshape((-1,1))
imp_f=SimpleImputer(strategy="most_frequent")
Embarked=imp_f.fit_transform(Embarked)

df.loc[:,"Age"]=result_2
df.loc[:,"Embarked"]=Embarked

print(df.info())

#Numpy和Pandas实现
data_=pd.read_csv("./datasets/Narrativedata.csv",index_col=0)

data_.loc[:,"Age"]=data_.loc[:,"Age"].fillna(data_.loc[:,"Age"].median())
data_.dropna(axis=0,inplace=True)
print("*"*20)
print(data_.info())

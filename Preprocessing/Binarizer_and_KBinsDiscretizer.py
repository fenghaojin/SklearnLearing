import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

df=pd.read_csv("./datasets/Narrativedata.csv",index_col=0)
Age=df.loc[:,"Age"].values.reshape((-1,1))
imp_median=SimpleImputer(strategy="median")
Age=imp_median.fit_transform(Age)
df.loc[:,"Age"]=Age
x=df.iloc[:,0].values.reshape(-1,1)

x=Binarizer(threshold=30).fit_transform(x)
est=KBinsDiscretizer(n_bins=3,encode="onehot",strategy="uniform")
x=est.fit_transform(x)

print(x.toarray())

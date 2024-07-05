from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df=pd.read_csv("./datasets/Narrativedata.csv",index_col=0)
print(df.info())

#填充缺失值
Age=df.loc[:,"Age"].values.reshape((-1,1))
imp_median=SimpleImputer(strategy="median")
Age=imp_median.fit_transform(Age)
Embarked=df.loc[:,"Embarked"].values.reshape((-1,1))
imp_f=SimpleImputer(strategy="most_frequent")
Embarked=imp_f.fit_transform(Embarked)
df.loc[:,"Age"]=Age
df.loc[:,"Embarked"]=Embarked
print("*"*30)
print(df.info())

x=df.iloc[:,1:3]
y=df.iloc[:,-1]#特征标签不允许一维

#编码
le=LabelEncoder()
le=le.fit(y)
label=le.transform(y)
print(le.classes_)
df.iloc[:,-1]=label
""" oe=OrdinalEncoder()
oe=oe.fit(x)
feature=oe.transform(x)
print(oe.categories_) """
oe=OneHotEncoder(categories="auto")
oe=oe.fit(x)
feature=oe.transform(x).toarray()
print(oe.get_feature_names_out())

newdf=pd.concat([df,pd.DataFrame(feature)],axis=1)
newdf.drop(["Sex","Embarked"],axis=1,inplace=True)
newdf.columns=['Age','Survived','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']
print(newdf.head())

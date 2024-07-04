from sklearn.preprocessing import StandardScaler

data=[[-1,2],[-0.5,6],[0,10],[1,18]]
print(type(data))

scaler=StandardScaler()
scaler=scaler.fit(data)
print(scaler.mean_)
print(scaler.var_)
data=scaler.transform(data)
print(data.mean())
print(data.std())

print(scaler.inverse_transform(data))

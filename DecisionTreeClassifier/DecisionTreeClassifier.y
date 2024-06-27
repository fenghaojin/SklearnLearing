import sklearn
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import graphviz

wine=load_wine()

Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)

test=[]
for i in range(10):
    clf=tree.DecisionTreeClassifier(criterion="entropy",
                                    random_state=30,
                                    splitter="random",
                                    max_depth=i+1,
                                    min_samples_leaf=10,
                                    min_samples_split=10
                                    )
    clf=clf.fit(Xtrain,Ytrain)
    score=clf.score(Xtest,Ytest)
    test.append(score)

print(test)
plt.plot(range(1,11),test,label="max_depth")
plt.legend()
plt.show()

feature_name=['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
#filled参数设置是否有颜色，rounded参数设置圆框还是方框
dot_data=tree.export_graphviz(clf,
                              feature_names=feature_name,
                              class_names=["琴酒","雪梨","贝尔摩德"],
                              filled=True,
                              rounded=True
                              )
graph=graphviz.Source(dot_data)
graph.view()

clf.feature_importances_
print([*zip(feature_name,clf.feature_importances_)])

score=clf.score(Xtrain,Ytrain)
print(score)

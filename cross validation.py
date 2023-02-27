import numpy as np
import pandas as pd

df=pd.read_csv('C://Users\hp\Downloads\heart_disease_data.csv')
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

y.value_count()

#holdout validation -train test split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,random_state=4)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
result=model.score(x_test,y_test)
result

#kfold cross validation with different random state 
from sklearn.model_selection import KFold
model=DecisionTreeClassifier()
kfold_validation=KFold(10)
from sklearn.model_selection import cross_val_score
result=cross_val_score(model,x,y,cv=kfold_validation)
result
print(np.mean(result))

#stratified k-fold cross validation
#if data is imbalanced
from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=5)
model=DecisionTreeClassifier()
scores=cross_val_score(model, x,y,cv=skfold)
print(np.mean(scores))

#leave one out cross val score
from sklearn.model_selection import LeaveOneOut
model=DecisionTreeClassifier()
leave_validation=LeaveOneOut()
result=cross_val_score(model, x,y,cv=leave_validation)
print(np.mean(result))

#leave p out
from sklearn.model_selection import LeavePOut
model=DecisionTreeClassifier()
lpo=LeavePOut(p=2)
result=cross_val_score(model,x,y,cv=lpo)
print(np.mean(result))





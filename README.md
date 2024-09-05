## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: swetha .M
RegisterNumber: 212223040223


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

![image](https://github.com/user-attachments/assets/4fbf8831-31ea-4d02-b18e-66eebdd75937)
![image](https://github.com/user-attachments/assets/84387970-b889-4fa9-9b2d-8efe671091f1)
![image](https://github.com/user-attachments/assets/704d1b99-93fb-4ad4-99f0-b7b25b615a2c)
![image](https://github.com/user-attachments/assets/8123670f-d25a-4d1a-8654-b7267e086faf)
![image](https://github.com/user-attachments/assets/4d8d4b7a-4e31-4a8f-85b0-78652e718539)
![image](https://github.com/user-attachments/assets/9ea1b5ca-103c-4494-abd5-4ce6b4c0d7b9)
![image](https://github.com/user-attachments/assets/77e06973-c7bb-4e9c-b78a-bd3e0167c38a)
![image](https://github.com/user-attachments/assets/800eb17f-1c5e-42be-97e6-2043de09296d)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

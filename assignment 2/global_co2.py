import pandas as pd
import numpy as np

dataset=pd.read_csv("global_co2.csv")


mean_value=dataset['Per Capita'].mean()
dataset['Per Capita']=dataset['Per Capita'].fillna(mean_value)

x=dataset.drop(["Per Capita"],axis="columns")
y=dataset["Per Capita"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)
a1=regressor.score(x_test,y_test)
print(a1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
a2=regressor.score(x_test,y_test)
print(a2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=0,test_size=0.3)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_train, y_train)
a3=lin_reg_2.score(x_test,y_test)
print(a3)
a=regressor.predict([[2011,5280,808,2199,2094,128,51]])
print(a)
b=regressor.predict([[2012,5965,937,2412,241,152,50]])
print(b)
c=regressor.predict([[2013,78,0,0.3,9,0,12]])
print(c)



y_pred=regressor.predict(x_test)
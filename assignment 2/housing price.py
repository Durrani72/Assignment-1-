import numpy as np
import pandas as pd

dataset=pd.read_csv("housing price.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)
y_pred=regressor.predict(x_test)
print("The Accuracuy of Linear Regression =",regressor.score(x_test,y_test))
print("Enter Housing ID=",regressor.predict([[1293]]))


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(x)
poly_reg.fit(X_poly, y)

x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=0,test_size=0.6)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_train,y_train)

print("The Accuracy of Polynomial Regression=",lin_reg_2.score(x_test,y_test))
print("Enter Housing ID=",lin_reg_2.predict(poly_reg.fit_transform([[1293]])))

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=15)
regressor.fit(x_train,y_train)
print("The Accuracuy of Decision Tree =",regressor.score(x_test,y_test))

y_pred=regressor.predict(x_test)


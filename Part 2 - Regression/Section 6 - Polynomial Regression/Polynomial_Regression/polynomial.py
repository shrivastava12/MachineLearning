# Polynomial Regression

# import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and test set
"""from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,_Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

# Visualising the Linear Regression results
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linear Regresssion)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# visualising the Ploynomial Regression for more smoothness and better curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with the Linear Regresssion

lin_reg.predict([[6.5]])


# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))






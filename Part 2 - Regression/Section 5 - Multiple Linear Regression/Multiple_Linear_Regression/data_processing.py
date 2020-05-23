# Multiple Linear Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# Encoding the Dependent variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x =  LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dumy variable Trap
X = X[:,1:]

# Splliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2 ,random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
# OrdinaryLeastSquare
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,2,4,5]]
# OrdinaryLeastSquare
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,4,5]]
# OrdinaryLeastSquare
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()


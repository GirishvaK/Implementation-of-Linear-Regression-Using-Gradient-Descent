# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
X = data.data[:, :3] # Features: 'MedInc', 'HouseAge', 'AveRooms'
Y = np.column_stack((data.target, data.data[:, 6])) # Targets: 'MedHouseVal', 'AveOccup'
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler() scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train) X_test = scaler_X.transform(X_test) Y_train = scaler_Y.fit_transform(Y_train) Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred) Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred) print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5]) # Print first 5 predictions
Developed by:Girishva.K 
RegisterNumber:25009292 
*/
```

## Output:
![523912663-b11ede2c-65db-4171-982c-23bfd4d5d603](https://github.com/user-attachments/assets/6aa3887d-ae85-45ab-8bd5-48fe5124678e)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

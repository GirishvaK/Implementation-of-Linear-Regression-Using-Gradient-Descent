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
import matplotlib.pyplot as plt

# Sample training data (Population of City, Profit)
X = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829])
y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233, 13.662])

# Number of samples
m = len(y)

# Add column of 1s for bias term
X_b = np.c_[np.ones((m, 1)), X]

# Initialize parameters
theta = np.zeros(2)

# Gradient Descent settings
alpha = 0.01
iterations = 1500

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Gradient descent algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta = theta - alpha * gradient
    return theta

# Train model
theta = gradient_descent(X_b, y, theta, alpha, iterations)

print("Theta values:", theta)

# Predict profit for any city population
population = float(input("Enter city population: "))
prediction = theta[0] + theta[1] * population
print("Predicted Profit:", prediction)

Developed by:Girishva.K 
RegisterNumber:25009292 
*/
```

## Output:

<img width="612" height="146" alt="522993289-0ba6bdc9-d8fc-4f5c-aaa1-e32f91efd56b" src="https://github.com/user-attachments/assets/e6e69ac9-ac38-4845-8979-ab305832fc6d" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

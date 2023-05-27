import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, X, y):
        # Add a column of ones to the feature matrix for the intercept term
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        # Calculate the coefficients using the normal equation
        X_transpose = np.transpose(X)
        X_transpose_dot_X = np.dot(X_transpose, X)
        inverse = np.linalg.inv(X_transpose_dot_X)
        self.coefficients = np.dot(np.dot(inverse, X_transpose), y)
    
    def predict(self, X):
        # Add a column of ones to the feature matrix for the intercept term
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        # Calculate the predicted values
        return np.dot(X, self.coefficients)
    
    def score(self, X, y):
        # Calculate the coefficient of determination (R-squared) score
        y_pred = self.predict(X)
        ssr = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ssr / sst)
        return r_squared
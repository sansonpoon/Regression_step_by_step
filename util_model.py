from typing import Optional, Union
import pandas as pd
import numpy as np
import scipy.stats

def hypothesis(X: pd.DataFrame, theta: np.ndarray) -> Union[pd.Series, list]:
    """
    Function to calculate y, given X and theta, using the hypothesis function.
    """
    y=0
    for value in zip(theta,X):
        y+=value[0]*X[value[1]]
    return y

#Define the cost function
def computeCost(X: pd.DataFrame, y: Union[pd.Series, list], theta: np.ndarray) -> float:
    """
    Define the cost function for gradient descent.
    """
    y1 = hypothesis(X, theta)
    return sum(np.sqrt((y1-y)**2))/(2*len(X))
    
def gradientDescent(X: pd.DataFrame, y: Union[pd.Series, list], theta: np.ndarray, alpha: float, i: int):
    """
    Function to do the gradient descent.
    """
    J = []  #cost function in each iterations
    k = 0
    while k < i:        
        y1 = hypothesis(X, theta)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/len(X))
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta
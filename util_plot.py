from typing import Optional, Union
import pandas as pd
import matplotlib.pylab as plt

def plot_cost(J: Union[pd.Series, list]) -> None:
    """
    Function to plot the gradient descent cost.
    """
    plt.figure()
    plt.scatter(x=list(range(0, len(J))), y=J)
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost')
    plt.show()
    return
    
def plot_compare(y: Union[pd.Series, list], y_hat: Union[pd.Series, list]) -> None:
    """
    Plot the true and predicted values
    """
    plt.figure()
    plt.scatter(x=list(range(0,len(y))),y= y, color='blue',marker='s')         
    plt.scatter(x=list(range(0,len(y_hat))), y=y_hat, color='cyan',marker='x')
    plt.ylabel('Example (n-th)')
    plt.ylabel('y')
    plt.legend(['True','Predicted'])
    plt.show()
    return
    
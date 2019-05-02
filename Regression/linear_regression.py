"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = np.sum(np.square(np.subtract(np.matmul(X,w),y)))/np.size(y,0)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)), np.matmul(X.transpose(),y))
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    xt_x= np.matmul(X.transpose(),X)
    e_v, vec= np.linalg.eig(xt_x)
    
    m = min(e_v)

    while(m<0.00001):
        xt_x = np.add(xt_x,np.multiply(0.1,np.identity(np.size(xt_x,0))))
        e_v, vec= np.linalg.eig(xt_x)
        m = min(e_v)
    
    w = np.matmul(np.linalg.inv(xt_x), np.matmul(X.transpose(),y))
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    xt_x= np.matmul(X.transpose(),X)
    
    xt_x = np.add(xt_x,np.multiply(lambd,np.identity(np.size(xt_x,0)))) 
    
    w = np.matmul(np.linalg.inv(xt_x), np.matmul(X.transpose(),y))
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    p=-19
    err_min=100
    bestlambda=None
    for i in range(39):
        lambd=10**(p+i)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        err = mean_square_error(w, Xval, yval)
        if(bestlambda is None):
            bestlambda = lambd
            err_min=err
        #print (lambd, err)
        elif(err<err_min):
            bestlambda = lambd
            err_min=err
    
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    lt=[]
    p=2
    col=[]
    while(p<=power):
        for j in range(len(X[0])):
            for i in range(len(X)):
                lt.append([X[i][j]**p])
            col.append(lt)
            lt=[]            
        p+=1
    #print(abc)
    
    for y in col:
        v=len(X[0]) 
        X=np.insert(X, [v], y, axis=1)
    return X



# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:02:42 2015
@author: priyanka


"""

import numpy as np
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt


def f(x):
    return np.sin(2 * np.pi * x)

# generate points used to plot
x_plot = np.linspace(0, 1, 100)[:, np.newaxis]

# generate points and keep a subset of them
n_samples = 100
X = np.random.uniform(0, 1, size=n_samples)[:, np.newaxis]
y = f(X) + np.random.normal(scale=0.3, size=n_samples)[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


def reg_ls(poly_order,regularization_constant):
    
    X=np.ones((X_train.shape[0],1+poly_order))
    x_plot_new=np.ones((x_plot.shape[0],1+poly_order)) # for plotting
    
    
    for i in range(poly_order+1): # Make basis
        X[:,i]=X_train[:,0]**i
        x_plot_new[:,i]=x_plot[:,0]**i 
    
    xtx = np.dot(X.T,X)
    xtx_inv = np.linalg.pinv(xtx + regularization_constant*np.eye(xtx.shape[0])) 
    xty = np.dot(X.T,y_train)
    
    w=np.dot(xtx_inv,xty) #parameters
    
    y_cap = np.dot(x_plot_new,w) #output
    return y_cap,w



    

def plot(poly_order,regularization_constant,title):
    
    y_cap,w=reg_ls(poly_order,regularization_constant)  
    plt.suptitle(title)  
    plt.subplot(211)
    plt.plot(x_plot, f(x_plot), color='green',label="Ideal fit")
    plt.plot(x_plot,y_cap,color='red',label="Least square fit")
    plt.scatter(X_train, y_train, s=10,label="Data")
    plt.ylim((-2, 2))
    plt.xlim((0, 1))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc="best", prop=dict(size=9))

    plt.subplot(212)
    plt.plot(w,label="Weight distribution")
    plt.ylabel('weights')
    plt.legend(loc="best", prop=dict(size=9))
    plt.show()


plot(15,0,"Non-regularized LS")
plot(15,6,"Regularized LS")

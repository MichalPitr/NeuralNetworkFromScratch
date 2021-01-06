#!/usr/bin/env python
# coding: utf-8

import numpy as np

class LossFunction(object):
    def forward(self, activation, y):
        pass
    
    def backward(self):
        pass
     
class MSELoss(LossFunction):
    def forward(self, activation, y):
        return np.mean((activation-y)**2)
    
    def backward(self, activation, y):
        return activation-y
    
class CrossEntropyLoss(LossFunction):
    """CrossEntropyLoss is assumed to be used after a sigmoid activation. 
    Softmax + Negative Log Likelihood will be added in the future"""
    def forward(self, y_hat, y):
        y_new = np.zeros(y_hat.shape)
        for i in range(y.shape[1]):
            y_new[y[0,i], i] = 1 

        return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    
    def backward(self, y_hat, y):
        return (y_hat-y)/(y_hat-y_hat*y_hat)


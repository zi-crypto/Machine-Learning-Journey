from math import e
import numpy as np

# Activation Functoin:RELU
def RELU(X):
    x = []
    for i in X:
        if i > 0:
            x.append(i)
        elif i <= 0:
            x.append(0) 
    return x

# RELU Activation Function can just be like that: 
"""
def RELU(X):
    x = []
    [x.append(max(0, i)) for i in X]
    return x
"""
# Activation Function:Sigmoid
def Sigmoid(X):
    x = []
    for i in X:
        x.append(float(1/(1+math.e**-i)))
    return x


# Activation Function:Step
def STEP(X):
    x = []
    for i in X:
        if i > 0:
            x.append(1)
        else:
            x.append(0)
    return x

# Activation Function:Softmax
def Softmax(X):
    exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

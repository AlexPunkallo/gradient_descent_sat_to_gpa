
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    m = y.size
    predictions = X.dot(theta).flatten()
    sqErrors = (predictions - y) ** 2
    J = (1.0 / (2 * m)) * sqErrors.sum()
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        predictions = X.dot(theta).flatten()
        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
        J_history[i, 0] = compute_cost(X, y, theta)
    return theta, J_history

data=pd.read_csv('training.csv',delimiter=',')
X = data["SAT"]
y = data["GPA"]

m = y.size
it = ones(shape=(m, 2)) 
it[:, 1] = X 

theta = zeros(shape=(2, 1))
iterations = 1500
alpha = 0.000000002

print 'The computational cost is:', compute_cost(it, y, theta)   

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print 'I valori di teta sono:'      
print theta			    

result = it.dot(theta).flatten()
plt.scatter(data["SAT"],data["GPA"],marker='o')
title('GPA/SAT')
xlabel('SAT Score')
ylabel('GPA Score')
plot(data["SAT"], result)
show()

from decimal import *
data=pd.read_csv('testfile.csv',delimiter=',')
X=data["SAT"]
vet=ones(X.shape[0])
y=theta[0][0]*vet+theta[1][0]*X
y=[float(Decimal("%.1f" % i)) for i in y]
y=pd.DataFrame({'ID':data.ix[:,0],'GPA':y})
y1=y.set_index('ID')
y1.to_csv('satgpa.csv')
print y

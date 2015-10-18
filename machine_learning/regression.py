import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
file_name = 'data/input/ex1data1.txt'
data = np.genfromtxt(file_name, dtype=float, delimiter = ',')

# hypothesis: function of x that has parameters in it.
# parameters: unknowns to be optimized.
# cost function: function of parameters (theta). the hypothesis and y values are related in a way such that
#  when minimizing theta the x values are as close to the y values in some sense.
def hypothesis(theta, x):
    theta[0] + theta_1[1]*x

# want to minimize hypothesis - y values
# minimize sum from 1 to size of training set (number of obs) of squared differences between
#  hypothesis(x_i) - y_i with respect to unknown parameters.

# linear regression with no intercept.
def cost_function(theta, x, y):
    return y.T.dot(y) - y.T.dot(x)*(theta) - theta*(x.T).dot(y) + theta * (x.T).dot(x)*(theta)

thetas = np.arange(0, 1, .01)
costs = [cost_function(theta, x, y) for theta in thetas]
plt.plot(thetas, costs)

# squared error cost function
# linear regression with intecept and slope.
def cost_function(theta, x, y):
    return y.T.dot(y) - y.T.dot(x).dot(theta) - theta.T.dot(x.T).dot(y) + theta.T.dot(x.T).dot(x).dot(theta)

x = data[:, 0]
x = x.reshape((len(x), 1))
y = data[:, 1]
y = y.reshape((len(y), 1))
theta = np.zeros(2)

X = np.append(np.ones(x.shape[0]).reshape(x.shape[0], 1), x, 1)

theta[0] = 1
theta[1] = 1
cost_function(theta, X, y)

theta1s = np.arange(0, 1, .01)
theta2s = np.ones(len(theta1s))

thetas = np.vstack((theta2s, theta1s))

costs = [cost_function(theta, X, y) for theta in thetas.T]

plt.plot(thetas[1, :], [cost.tolist()[0][0] for cost in costs])

def gradient_descent(theta, X, y, learning_rate, niter):
    nobs = y.shape[0]
    cost_function_outcomes = np.zeros(niter, 1)



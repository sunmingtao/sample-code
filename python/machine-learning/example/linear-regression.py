import numpy as np  # NumPy is the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt  # Plot data
from sklearn.linear_model import LinearRegression  # Linear regression model


X = np.array([i+1 for i in range(100)])  #[1, 2, 3, 4, .....100 ]
y = np.array([(i+1) * 2+ 30 for i in range(100)]) # y = 2 * x + 30, [33, 36, 39, ...., 330]
noise = np.random.normal(0, 10, 100) # An array of 100 Random numbers between -10 and 10
y_noise = y + noise  # Add  some noise
plt.axis([0, 100, 0, 250]) # Set the range of X axis and y axis
plt.plot(X, y_noise, 'bo') # Plotted as blue dots
plt.xlabel('X')
plt.ylabel('y')

X = X.reshape(100, 1)  # Reshape from vector to matrix of shape (100, 1), it's mandated by the fit() method below
y = y.reshape(100, 1)  # Reshape from vector to matrix of shape (100, 1), it's mandated by the fit() method below
regression = LinearRegression(normalize=True) # Instantiate a LinearRegression object
regression.fit(X, y)  # Search for the best parameter

# Test the parameter
X_test = np.array([i+1 for i in range(100)]) # [1, 2, 3, 4, .....100 ]
y_pred = regression.predict(X_test.reshape(-1,1))

plt.plot(X_test, y_pred, 'r') # Plotted as red line
plt.xlabel('X')
plt.ylabel('y')

# Print out a, b in y = ax + b
print('y = {} * x + {}'.format(regression.coef_[0][0], regression.intercept_[0]))

X_test = np.array([i+1 for i in range(100)]) # [1, 2, 3, 4, .....100 ]
y_pred = np.array([(i+1) * 3 + 20 for i in range(100)])

plt.plot(X_test, y_pred, 'r') # Plotted as red line
plt.xlabel('X')
plt.ylabel('y')
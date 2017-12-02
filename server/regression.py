import numpy as np
import matplotlib.pyplot as plt 
data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)
data_x /= np.max(data_x)

plt.scatter(data_x,data_y)
plt.show()
data_x = np.hstack((np.ones_like(data_x), data_x))

order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, np.power(error, 2)

w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-9

# Perform Gradient Descent
iterations = 1
while True:
    gradient, error = get_gradient(w, train_x, train_y)
    new_w = w - alpha * gradient
    
    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print "Converged."
        break
    
    # Print error every 50 iterations
    if iterations % 100 == 0:
        print iterations, error
    
    iterations += 1
    w = new_w

X=[]
Y=[]
for x in data_x:
	X.append(x[1])
	Y.append(w[0]+w[1]*x[1])


plt.scatter(data_x[:,1],data_y)
plt.plot(X,Y)
plt.show()

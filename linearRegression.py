import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2, 3, 5, 7, 10, 11, 12, 13], #just created my own data
              [1, 1, 1, 1, 1, 1, 1, 1, 1]])
y = np.array([5, 11, 14, 25, 36, 39, 45, 50, 54])#plt.show()
thetas = np.random.rand(1,2) #initialize parameters randomly

#batch gradient descent (we use the entire dataset in each iteration)
m = y.size
for i in range(2000):
    y_hat = np.dot(thetas, X) #get the predictions
    thetas = thetas - 0.01*(1.0/m)*np.dot((y_hat - y), X.T)#subtract the gradient of the loss, multiplied by alpha


plt.scatter(X[0], y)
#plot the linear regression line we got
testX = np.linspace(0,15,100)
testY = thetas[0][0]*testX+thetas[0][1]
plt.plot(testX, testY, '-r')

#plot the linear regresssion line from Excel
trueX = np.linspace(0,15,100)
trueY = 3.8888*trueX + 3.3462
plt.plot(trueX, trueY, '-r', color='g')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

print(thetas)

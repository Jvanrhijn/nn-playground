import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)

# CS231n spiral dataset
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N*K, D))  # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

num_hidden = 2
neurons_per_hidden = 50
epochs = 250

learn_rate = 1e-3
window = 0.9
window_sq = 0.999

network = net.NeuralNetwork(D, K, num_hidden, neurons_per_hidden,
                            activation='relu',
                            cost='ce',
                            h_et_al=True)

acc_before = network.validate(X, y)

costs = network.train(X, y, epochs,
                      optimizer='nadam',
                      lr=learn_rate,
                      beta1=window,
                      beta2=window_sq,
                      gamma=0.9,
                      quiet=False, save=True, reg=1e-4)

acc_after = network.validate(X, y)
print("Accuracy before: {0}\nAccuracy after: {1}\n".format(acc_before, acc_after))

figure = plt.figure()
ax_data = figure.add_subplot(121)
ax_cost = figure.add_subplot(122)

ax_cost.semilogy(costs)
ax_cost.set_ylabel("Cost function")
ax_cost.set_xlabel("Epoch")

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
points = np.c_[xx.ravel(), yy.ravel()]
results = []
for idx, point in enumerate(points):
    results.append(np.argmax(network.forward_pass(point)))
results = np.array(results).reshape(xx.shape)

ax_data.contourf(xx, yy, results, cmap=plt.cm.Spectral, alpha=0.8)
ax_data.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
ax_cost.grid()

plt.show()

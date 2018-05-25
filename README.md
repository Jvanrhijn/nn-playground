# nn-playground

Neural network implemented in Python + NumPy, mostly to teach myself about how these things work and to learn more about different convex optimization algorithms.

**Usage**

Import src.network, src.models, src.layers and src.optim. Create your `training_data` and `training_answers`. Choose which algorithm and which type of activation function to use, then create the network:

~~~python
network = src.network.NeuralNet(input_size, output_size, num_hidden, neurons_per_hidden, activation='relu', cost='ce')
~~~

Now, train your network on the input data:

~~~python
costs = network.train(training_data, training_answers, epochs, optimizer='momentum', quiet=False, save=True,
                      optimizer='momentum', lr=1e-3, mom=0.9, nesterov=True)
~~~

The network will now use stochastic gradient descent with Nesterov's algorithm. If you set `quiet=False`, the trainer will print the cost function for the whole data set after each training epoch. The `save` kwarg saves the cost function for each epoch. Finally, run the network on any other data point:

~~~python
result = network.forward_pass(some_input)
~~~

**Examples**

Some examples are included which show how to use the API. Below example demonstrates the convergence of several included optimization algorithms (hyperparameters tuned manually).

[![optimization demo](https://i.imgur.com/rKsksFW.png)](https://i.imgur.com/rKsksFW.png)

The below example shows the network's classification capabilities, after being trained on CS231n's spiral dataset. This network consisted of a two hidden layers with 50 ReLu activated neurons, and was optimized with Nadam for a cross-entropy cost function with L2 regularization. The shaded regions indicate the decision boundaries.

[![classification demo](https://i.imgur.com/FH0oLIF.png)](https://i.imgur.com/FH0oLIF.png)

**Dependencies**

* Python 3.x
* Matplotlib
* NumPy


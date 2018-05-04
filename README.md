# nn-playground

Neural network implemented in Python + NumPy, mostly to teach myself about how these things work and to learn more about different convex optimization algorithms.

**Usage**

Import src.network, src.models, src.layers and src.optim. Create your `training_data` and `training_answers`. Choose which algorithm and which type of activation function to use, then create the network and optimizer:

~~~python
network = src.network.NeuralNet(input_size, output_size, num_hidden, neurons_per_hidden, ly.TanhLayer, mod.mse)
nag_optim = src.optim.NAGOptimizer(learn_rate, momentum_parameter, network)
~~~

This network has a tanh activation function, and uses Nesterov's accelerated gradient descent algorithm to optimize a mean square error cost function. Now, train your network on the input data:

~~~python
costs = network.train(training_data, training_answers, epochs, nag_optim, quiet=False, save=True)
~~~

If you set `quiet=True`, the trainer will print the cost function for the whole data set after each training epoch. The `save` kwarg saves the cost function for each epoch. Finally, run the network on any other data point:

~~~python
result = network.forward_pass(some_input)
~~~

**Examples**

Some examples are included which show how to use the API. One example demonstrates the network's ability to perform certain tasks (at the moment, approximating a single-valued function on a given interval). The other shows the difference in performance when using different convex optimization algorithms (currently only stochastic gradient descent and Nesterov's accelerated gradient descent). The below figures show the network's attempt at fitting a Gaussian function of standard deviation 0.25 and zero mean, usig stochastic GD and Nesterov GD. The cost function is a mean square error cost.

[![optimization demo](https://i.imgur.com/FlU5sj1.png)](https://i.imgur.com/FlU5sj1.png)

**Dependencies**

* Python 3.x
* Matplotlib
* NumPy


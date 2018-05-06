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

If you set `quiet=False`, the trainer will print the cost function for the whole data set after each training epoch. The `save` kwarg saves the cost function for each epoch. Finally, run the network on any other data point:

~~~python
result = network.forward_pass(some_input)
~~~

**Examples**

Some examples are included which show how to use the API. One example demonstrates the network's ability to perform certain tasks (at the moment, approximating a single-valued function on a given interval). The other shows the difference in performance when using different convex optimization algorithms (currently only stochastic gradient descent and Nesterov's accelerated gradient descent). The below figures show the network's attempt at fitting a Gaussian function of standard deviation 0.25 and zero mean, usig stochastic GD and Nesterov GD. The cost function is a mean square error cost.

[![optimization demo](https://imgur.com/fkei4pO.png)](https://imgur.com/fkei4pO.png)

The below example shows the network's classification capabilities. It was given 100 randomly generated points, each assigned a color - red if it lies above a given line, blue if below. The figure below shows the result of running the network on a set of 1000 randomly generated points of validation data. This network consisted of a single hidden layer with 1000 ReLu activated neurons, and was optimized with Nesterov's algorithm for an SVM cost function. The resulting accuracy on the valdation set was 97.1%, versus 63.1% before training (corresponging to the area below the separation line).

[![classification demo](https://imgur.com/Rv8lZR4.png)](https://imgur.com/Rv8lZR4.png)

**Dependencies**

* Python 3.x
* Matplotlib
* NumPy


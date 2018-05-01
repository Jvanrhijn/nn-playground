import collections
import numpy as np


Activator = collections.namedtuple('Activator', ['activator', 'gradient'])


re_lu = Activator(lambda x: np.maximum(0, x), lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda y: 0, lambda y: 1]))


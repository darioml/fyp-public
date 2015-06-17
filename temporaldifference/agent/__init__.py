__author__ = 'darioml'

import numpy as np


# These are a set of helper functions!


# softmax function
def softmax(w, t = .1):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    # print dist
    return dist

#pick based on a random distribution
def pick_from_distribution(dist):
    index = 0
    rand = np.random.uniform(0, 1)

    for key, prob in enumerate(dist):
        index += prob
        if index >= rand:
            return key

    return prob[-1]  # If we're still here, send the last one. Happes because of floating point

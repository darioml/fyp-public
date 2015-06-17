__author__ = 'darioml'

import numpy as np
import backprop__ as helper



ITERATION_SIZE = [10, 20, 50, 200, 500, 800, 1000, 1500, 2000]
HIDDEN_NODES = 25
NO_ITERATIONS = 7

labels, data = helper.get_data(filename='../data/mat/ball_with_speed.mat', flatten=False)
labels = labels[0,:].flatten()

results = np.zeros((len(ITERATION_SIZE), NO_ITERATIONS))
times = np.zeros((len(ITERATION_SIZE), NO_ITERATIONS))

for i in range(len(ITERATION_SIZE)):
    _,_,results[i,:],times[i,:] = helper.test_simple_backprop(data, labels, HIDDEN_NODES, NO_ITERATIONS, ITERATION_SIZE[i], debug=True)


helper.save('backprop_2img_dir_iters', accuracy=results, times=times,
            nodes=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE)

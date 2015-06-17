__author__ = 'darioml'

import numpy as np
import backprop__ as helper



ITERATION_SIZE = 500
HIDDEN_NODES = [15,17,19,21,23,25,27,29,31,33,35,40,45]
NO_ITERATIONS = 7

labels, data = helper.get_data(filename='../data/mat/ball_with_speed.mat', flatten=False)
labels = labels[0,:].flatten()

results = np.zeros((len(HIDDEN_NODES), NO_ITERATIONS))
times = np.zeros((len(HIDDEN_NODES), NO_ITERATIONS))

for i in range(len(HIDDEN_NODES)):
    _,_,results[i,:],times[i,:] = helper.test_simple_backprop(data, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)


helper.save('backprop_2img_dir_nodes', accuracy=results, times=times,
            index=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE)

__author__ = 'darioml'

import numpy as np
import backprop__ as helper



ITERATION_SIZE = 500
HIDDEN_NODES = [15,17,19,21,23,25,27,29,31,33,35,40,45]
NO_ITERATIONS = 7

labels, data = helper.get_data(filename='../data/mat/ball_with_speed.mat', flatten=False)

real_labels = np.zeros( len(labels.T), int )
real_labels[ (labels[0,:] == 0) & (labels[1,:] == 3) ] = 0
real_labels[ (labels[0,:] == 1) & (labels[1,:] == 3) ] = 1
real_labels[ (labels[0,:] == 2) & (labels[1,:] == 3) ] = 2
real_labels[ (labels[0,:] == 3) & (labels[1,:] == 3) ] = 3
real_labels[ (labels[0,:] == 0) & (labels[1,:] == 5) ] = 4
real_labels[ (labels[0,:] == 1) & (labels[1,:] == 5) ] = 5
real_labels[ (labels[0,:] == 2) & (labels[1,:] == 5) ] = 6
real_labels[ (labels[0,:] == 3) & (labels[1,:] == 5) ] = 7
real_labels[ (labels[0,:] == 0) & (labels[1,:] == 8) ] = 8
real_labels[ (labels[0,:] == 1) & (labels[1,:] == 8) ] = 9
real_labels[ (labels[0,:] == 2) & (labels[1,:] == 8) ] = 10
real_labels[ (labels[0,:] == 3) & (labels[1,:] == 8) ] = 11

labels = real_labels.flatten()

results = np.zeros((len(HIDDEN_NODES), NO_ITERATIONS))
times = np.zeros((len(HIDDEN_NODES), NO_ITERATIONS))

for i in range(len(HIDDEN_NODES)):
    _,_,results[i,:],times[i,:] = helper.test_simple_backprop(data, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)


helper.save('backprop_2img_both_nodes', accuracy=results, times=times,
            nodes=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE)

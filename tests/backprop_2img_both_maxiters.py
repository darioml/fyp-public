__author__ = 'darioml'

import numpy as np
import backprop__ as helper



ITERATION_SIZE = [10, 20, 50, 200, 500, 800, 1000, 1500, 2000]
HIDDEN_NODES = 25
NO_ITERATIONS = 7

labels, data = helper.get_data(filename='../data/mat/ball_with_speed.mat', flatten=False)

real_labels = np.zeros( len(labels.T),int )
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

real_labels = real_labels.flatten()

labels = real_labels

results = np.zeros((len(ITERATION_SIZE), NO_ITERATIONS))
times = np.zeros((len(ITERATION_SIZE), NO_ITERATIONS))

for i in range(len(ITERATION_SIZE)):
    _,_,results[i,:],times[i,:] = helper.test_simple_backprop(data, labels, HIDDEN_NODES, NO_ITERATIONS, ITERATION_SIZE[i], debug=True)


helper.save('backprop_2img_both_iters', accuracy=results, times=times,
            nodes=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE)

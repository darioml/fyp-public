__author__ = 'darioml'

import numpy as np
import backprop__ as helper






ITERATION_SIZE = 500
HIDDEN_NODES = [15,17,19,21,23,25,27,29,31,33,35]
NO_ITERATIONS = 20

labels, data = helper.get_data()

results = np.zeros(len(HIDDEN_NODES))
times = np.zeros(len(HIDDEN_NODES))

for i in range(len(HIDDEN_NODES)):
    results[i],times[i],_,_ = helper.test_simple_backprop(data, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)

helper.save('backprop_hiddennodes', accuracy=results, times=times,
            index=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE)

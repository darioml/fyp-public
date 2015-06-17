__author__ = 'darioml'

import numpy as np
import backprop__ as helper




ITERATION_SIZES = [10, 20, 50, 200, 500, 800, 1000, 1500, 2000]
NO_HIDDEN_NODES = 20
NO_ITERATIONS = 20



labels, data = helper.get_data()

results = np.zeros(len(ITERATION_SIZES))
times = np.zeros(len(ITERATION_SIZES))

for i in range(len(ITERATION_SIZES)):
    results[i],times[i],_,_ = helper.test_simple_backprop(data, labels, NO_HIDDEN_NODES, NO_ITERATIONS, ITERATION_SIZES[i], debug=True)

helper.save('backprop_maxiter', accuracy=results, times=times,
            index=ITERATION_SIZES, no_iteration=NO_ITERATIONS, no_hidden_nodes=NO_HIDDEN_NODES)
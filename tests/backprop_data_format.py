__author__ = 'darioml'


import numpy as np
import backprop__ as helper


HIDDEN_NODES = [15,17,19,21,23,25,27,29,31,33,35]
NO_ITERATIONS = 12
ITERATION_SIZE = 500

labels, data = helper.get_data()



data2 = data*255
data3 = (2*data)-1


results = np.zeros((3,len(HIDDEN_NODES)))
times = np.zeros((3,len(HIDDEN_NODES)))

for i in range(len(HIDDEN_NODES)):
    results[0,i],times[0,i],_,_ = helper.test_simple_backprop(data, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)
    results[1,i],times[1,i],_,_ = helper.test_simple_backprop(data2, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)
    results[2,i],times[2,i],_,_ = helper.test_simple_backprop(data3, labels, HIDDEN_NODES[i], NO_ITERATIONS, ITERATION_SIZE, debug=True)


helper.save('backprop_data_format', accuracy=results, times=times,
            index=HIDDEN_NODES, no_iteration=NO_ITERATIONS, max_iteration=ITERATION_SIZE,
            legend=['[0,1]', '[0, 255]', '[-1,1]'])
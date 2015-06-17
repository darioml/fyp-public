__author__ = 'darioml'

import numpy as np
import backprop__ as helper
from backprop.scipy_optim import NN_1HL
from sklearn import cross_validation
from sklearn.metrics import accuracy_score,confusion_matrix


# Using the best metrics, let's test a few times
HIDDEN_NODES = 25
ITERATION_SIZE = 800


NUMBER_OF_TESTS = 5

labels, data = helper.get_data()


accuracy_accumulator = 0.0;

y_final_pred = []
y_final_test = []

for i in range(NUMBER_OF_TESTS):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.2)
    nn = NN_1HL(maxiter=ITERATION_SIZE, hidden_layer_size=HIDDEN_NODES)
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)

    accuracy_accumulator += accuracy_score(y_test, y_pred)

    # save results
    if y_final_pred is []:
        y_final_pred = y_pred
        y_final_test = y_test
    else:
        y_final_pred = np.concatenate( (y_final_pred, y_pred) )
        y_final_test = np.concatenate( (y_final_test, y_test) )

    print '  Done with %i of %i' % (i, NUMBER_OF_TESTS)


helper.save('backprop_simple_confmatrix', y_pred=y_pred, y_test=y_test,
            hidden_nodes=HIDDEN_NODES, iter_size=ITERATION_SIZE, no_tests=NUMBER_OF_TESTS)
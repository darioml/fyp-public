__author__ = 'darioml'

# Common library for backprop testing
from backprop.nn_scipy_opti import NN_1HL
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint
import scipy.io
import time
import datetime
import os

def get_data(filename='../data/mat/ball.mat', flatten=True):
    ## Let's get our data

    data_file = scipy.io.loadmat(filename)

    data = np.array(data_file['X'])
    labels = np.array(data_file['Y'], 'uint8').T

    if flatten is True:
        labels = labels.flatten()

    return labels, data/255



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(4)
    plt.xticks(tick_marks, range(4)) #, rotation=45
    plt.yticks(tick_marks, range(4))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def debug_print(debug, statement):
    if debug is True:
        print statement

def test_simple_backprop(data, labels, hidden_nodes, iterations, maxiter=200, debug=False, plot=False):
    times = []
    accuracy = []

    debug_print(debug, 'Starting iteration - Nodes: %i, Reps: %i, Max Iters: %i' % (hidden_nodes, iterations, maxiter))

    for i in range(iterations):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.2)
        nn = NN_1HL(maxiter=maxiter, hidden_layer_size=hidden_nodes)

        time_now = time.time()
        nn.fit(X_train, y_train)
        times.append( time.time() - time_now )

        accuracy.append(accuracy_score(y_test, nn.predict(X_test)))

        debug_print(debug, '  Finished %i of %i' % (i, iterations))

    return np.mean(accuracy),np.mean(times),accuracy,times


def save(filename, __file_append=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') + "_", **kwargs):
    scipy.io.savemat('../results/' + __file_append + filename, kwargs)


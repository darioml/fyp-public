import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pprint import pprint
from basic.perception import Perception, Trainer
from basic.bp_neuron import Network

# this doens't actually work --

test = Network(2,4)

for i in range(1000):
    rands = np.random.rand(2)
    inputs = [0, 0]
    if rands[0] < 0.5:
        inputs[0] = 0
    else:
        inputs[0] = 1

    if rands[1] < 0.5:
        inputs[1] = 0
    else:
        inputs[1] = 1

    ans = 1 if inputs[0] == inputs[1] else 0
    print inputs, ans
    test.train(inputs, ans)


print
print
print
print "This should be 1"
print test.feedforward([1,1])
print test.feedforward([-1,-1])
print "This should be -1"
print test.feedforward([-1,1])
print test.feedforward([1,-1])

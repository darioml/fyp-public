__author__ = 'darioml'

import numpy as np
from pprint import pprint
from perception import Perception as perc_basic


class Perception(perc_basic):
    def activate(self, localSum):
        return 1.0 / (1.0 + np.exp(-localSum));

    def updateWeight(self, index, delta):
        self.weights[index] += 0.5 * delta;


class Network:
    def __init__(self, inputSize, hiddenSize):
        self.neurons = [[], [], []]

        for i in range(hiddenSize):
            perc = Perception(inputSize + 1)
            self.neurons[1].append(perc)

        perc = Perception(hiddenSize)
        self.neurons[2].append(perc)

    def feedforward(self, inputs):
        inputs = inputs + [1]
        hiddenOut = []

        for node in self.neurons[1]:
            hiddenOut.append(node.feedforward(inputs))

        return self.neurons[2][0].feedforward(hiddenOut)

    def train(self, inputs, answer):
        inputs = inputs + [1]

        hiddenOut = []
        for node in self.neurons[1]:
            hiddenOut.append(node.feedforward(inputs))
        result = self.neurons[2][0].feedforward(hiddenOut)

        # 1. Ouput Delta
        deltaOutput = result * (1 - result) * (answer - result)

        # 2. Output Weights
        index = 0
        for node in self.neurons[1]:
            change = deltaOutput * hiddenOut[index]
            # print index, deltaWeight, self.neurons[2][0].weights
            self.neurons[2][0].updateWeight(index, change)
            index += 1;

        # Now, do the same for the nodes from input to output
        index = 0
        for node in self.neurons[1]:
            error = deltaOutput * self.neurons[2][0].weights[index]
            dHidden = error * hiddenOut[index] * (1 - hiddenOut[index])

            # now adjust the weight for the second layer.
            for key, value in enumerate(inputs):
                change = dHidden * value
                self.neurons[1][index].updateWeight(key, change)

            index += 1

    def showWeights(self):
        for layer in self.neurons:
            for node in layer:
                print node.weights
        print

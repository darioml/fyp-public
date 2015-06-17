import numpy as np
from pprint import pprint


class Perception:
    def __init__(self, n):
        self.learningRate = 0.5
        self.weights = 2 * np.random.rand(n) - 1

    def feedforward(self, invars):
        localSum = 0;
        for i in range(len(invars)):
            localSum += invars[i] * self.weights[i]
        return self.activate(localSum)

    def activate(self, localSum):
        return 1 if localSum > 0 else 0;

    def train(self, invars, desired):
        myGuess = self.feedforward(invars)
        error = desired - myGuess

        for i in range(len(self.weights)):
            self.weights[i] += self.learningRate * error * invars[i];


class Trainer:
    def __init__(self, grad, offset):
        self.grad = grad
        self.offset = offset

    def train_2d(self, perception, iterations):
        for iteration in range(iterations):
            inputs = np.random.rand(len(perception.weights))
            answer = 1 if inputs[1] > (inputs[0] * self.grad + self.offset) else 0
            perception.train(inputs, answer)

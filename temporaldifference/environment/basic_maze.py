__author__ = 'darioml'

## this is a basic maze environment for TD learning
import operator
import numpy as np
import math

class environment(object):
    def __init__(self, size):
        self.size = int(size)

        self.reset()

    def reset(self):
        self.current_location = (self.size/2,self.size/2)

    def set_rewards(self, locations):
        self.rewards = locations;

    def get_possible_actions(self, current_location=None):
        if current_location is None:
            current_location = self.current_location

        x,y = current_location

        possible_actions = [(0,0)]
        if x > 0:
            possible_actions.append((-1,0))
        if x < self.size-1:
            possible_actions.append((1,0))
        if y > 0:
            possible_actions.append((0,-1))
        if y < self.size-1:
            possible_actions.append((0,1))

        return possible_actions


    def take_action(self, action, direction, end, current_state=None):
        if current_state is None:
            current_state = self.current_location
        reward = 0
        next_state = tuple(map(operator.add, current_state, action))

        if end is True:
            if next_state == self.rewards[direction]:
                reward = 5
            else:
                reward = 0
            # if direction == 0 and next_state == (0,2):
            #     reward = 5
            # elif direction == 1 and next_state == (4,2):
            #     reward = 5
            # else:
            #     reward = 0

        self.update_state(next_state)
        return next_state, reward

    def update_state(self, state):
        self.current_location = state

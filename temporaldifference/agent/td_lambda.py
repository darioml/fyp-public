__author__ = 'darioml'

import numpy as np
import agent

# TD(lamda) algorithm using matrix to store V_t
class table(object):
    def __init__(self, env, direction, alpha, gamma, ld):
        self.alpha = alpha
        self.gamma = gamma
        self.ld = ld

        self.V = np.zeros((direction,env.size,env.size))
        self.e = np.zeros((direction,env.size,env.size))

    def eval(self, d, x, y):
        return self.V[d,x,y]

    def eval_actions(self, possible, d, x, y):
        evaluations = []
        for d_x,d_y in possible:
            evaluations.append(self.eval( d, x+d_x, y+d_y ))
        return evaluations

    def update_estimators(self,d,x,y,reward,new_loc):
        current_eval = self.eval(d, x, y)
        next_eval = self.eval(d, new_loc[0], new_loc[1])
        delta = reward + self.gamma*next_eval - current_eval

        self.e[d,x,y] += 1
        self.e[d,:,:] *= self.gamma * self.ld

        self.V[d,:,:] += self.alpha * delta * self.e[d,:,:]

    def take_action(self, possible_actions, evaluations):
        softmax = agent.softmax(evaluations)
        action_idx = agent.pick_from_distribution( softmax )
        return possible_actions[action_idx]

__author__ = 'darioml'

# this is just a test to get the principals correct.

import numpy as np
import operator



V = np.zeros((5,5))
# V[0,0] = 10
print V


current_location=(1,2)

def get_possible_actions(current_location):
    x,y = current_location
    if current_location == (0,0):
        return [(4,4)]

    possible_actions = []
    if x > 0:
        possible_actions.append((-1,0))
    if x < 4:
        possible_actions.append((1,0))
    if y > 0:
        possible_actions.append((0,-1))
    if y < 4:
        possible_actions.append((0,1))

    return possible_actions

def softmax(w, t = .5):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    # print dist
    return dist

def random_distr(l):
    r = np.random.uniform(0, 1)
    s = 0
    for item, prob in enumerate(l):
        s += prob
        if s >= r:
            return item
    return l[-1]  # Might occur because of floating point inaccuracies


for i in range(10000):
    print current_location,
    x,y = current_location
    possible_action = get_possible_actions(current_location)
    # print possible_action

    pi = []
    for d_x,d_y in possible_action:
        pi.append(V[x+d_x,y+d_y])
    #
    # print pi
    # print softmax(pi)
    # print random_distr(softmax(pi))
    movement = int(random_distr(softmax(pi)))

    new_loc = tuple(map(operator.add, current_location, possible_action[movement]))

    print 'move in (%i %i)' % possible_action[movement],
    print 'new loc: (%i %i)' % new_loc,

    reward = -.05;
    if possible_action == [(4,4)]:
        reward = 1

    old_val = V[current_location[0],current_location[1]]
    V[current_location[0],current_location[1]] += 0.1*(reward + 1*V[new_loc[0],new_loc[1]] - V[current_location[0],current_location[1]])

    print 'updated V[%i,%i] from %f to %f' % (current_location[0], current_location[1], old_val, V[current_location[0],current_location[1]])

    current_location = new_loc

print V
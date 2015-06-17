__author__ = 'darioml'


import numpy as np
import operator



V = np.zeros((2,5,5))

def get_possible_actions(current_location):
    x,y = current_location

    possible_actions = [(0,0)]
    if x > 0:
        possible_actions.append((-1,0))
    if x < 4:
        possible_actions.append((1,0))
    if y > 0:
        possible_actions.append((0,-1))
    if y < 4:
        possible_actions.append((0,1))

    return possible_actions

def softmax(w, t = .1):
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



def debug_log(statement, print_me=False, hold_line=True):
    if print_me is True:
        print statement,
        if hold_line is False:
            print


def take_action(current_state, action, iteration):
    reward = 0
    next_state = tuple(map(operator.add, current_location, possible_action[movement]))

    if iteration == 9:
        if next_state == (0,2):
            reward = 1

    return next_state, reward




for j in range(500):
    # reset location
    current_location=(2,2)

    # ball direction
    direction = 0
    debug_log(direction)

    for i in range(10):
        debug_log(current_location)
        x,y = current_location
        possible_action = get_possible_actions(current_location)
        # print possible_action

        pi = []
        for d_x,d_y in possible_action:
            pi.append(V[direction,x+d_x,y+d_y])
        #
        # print
        # print possible_action
        # print pi
        # print softmax(pi)
        movement = int(random_distr(softmax(pi)))
        # print movement

        new_loc,reward = take_action(current_location, movement, i)

        debug_log( 'move in (%i %i)' % possible_action[movement])
        debug_log( 'new loc: (%i %i)' % new_loc)

        old_val = V[direction,current_location[0],current_location[1]]
        V[direction,current_location[0],current_location[1]] += 0.1*(reward + .9*V[direction,new_loc[0],new_loc[1]] - V[direction,current_location[0],current_location[1]])

        debug_log( 'updated V[%i,%i] from %f to %f' % (current_location[0], current_location[1], old_val, V[direction,current_location[0],current_location[1]]))

        if reward > 0:
            print 'GOOD'
        elif i==9:
            print 'BAD, %i %i ' % new_loc
        debug_log('', hold_line=False)
        current_location = new_loc

np.set_printoptions(linewidth=90)
print V
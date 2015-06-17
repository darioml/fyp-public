import numpy as np
import scipy.io
import imp

import matplotlib.pyplot as plt

N1N = imp.load_source('*', '../backprop/scipy_optim.py')


# Single Dimension Temporal Difference
#   based on the up or down motion of the ball, this algorithm should move up or down on a given field.
#   A reward is given only on achieving the final location.


#   Ball Motion UP            BALL MOTION DOWN
#   ####                      ####
#   #  #                      #+1#
#   ####                      ####
#   #  #                      #  #
#   ####                      ####
#   #  #                      #  #
#   ####                      ####
#   #  # <- Start Location -> #  #
#   ####                      ####
#   #  #                      #  #
#   ####                      ####
#   #  #                      #  #
#   ####                      ####
#   #+1#                      #  #
#   ####                      ####


# Let's prepare our data. We will show a (random) sequence of 7 images. Based off that, the agent should decide what
# direction to move in

############
# PARAMS
############
field_size = 7
image_number = 7

alpha = 0.1
ld = 0.7

episodes = 10000

############
# SETUP
############
field = np.zeros(7)



############
# DATA
############

data = scipy.io.loadmat('../data/ball.mat')

labels = data['Y']
data = data['X']


images = data[(0,20,40,60,80,100,120),:].T
images = images/255


def evaulate_current(input, weight_in, weight_hidden):
    one = np.array(1).reshape(1,)
    hidden_sum = np.dot(weight_in,np.concatenate((input, one),1).T)
    hide = 1./(1 + np.exp(-hidden_sum))

    output_sum = np.dot(weight_hidden, np.concatenate((hide, one),1).T)
    return 1./(1 + np.exp(-output_sum))

def back_prop(weight_in, weight_hidden, e_in, e_hidden, reward, current_output, next_output, next_input, alpha, ld):
    one = np.array(1).reshape(1,)

    next_input_bias = np.concatenate((next_input, one),1)

    # update the hidden to output layer weight
    weight_hidden = weight_hidden + alpha * (next_output - current_output) * e_hidden

    # update the input to hidden layer weight
    weight_in = weight_in + alpha * (next_output - current_output) * e_in

    next_output_real = evaulate_current(next_input,weight_in,weight_hidden)

    # find the hidden layer output i.e. hide

    hidden_sum = np.dot(weight_in, next_input_bias.T)
    hide_input = 1 / (1 + np.exp(-hidden_sum))
    hide_input_biased = np.concatenate((hide_input, one),1)

    # update hidden to output layer eligibility trace
    e_hidden = ld * e_hidden + (1 - next_output_real) * next_output_real * hide_input_biased

    # update input to hidden layer eligibility trace
    n_i_b = np.concatenate((next_input,one),1)

    e_B = np.dot((((1 - hide_input)*hide_input) * weight_hidden[:,0:-1]).T, n_i_b.reshape((1,len(n_i_b))))
    e_in = ld * e_in + ((1 - next_output_real) * next_output_real * e_B)

    return weight_in, weight_hidden, e_in, e_hidden


# https://github.com/stober/td/blob/master/bin/td_example.py
def softmax(w, t = 1.0):
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


def picksoftmax(w, t=1.0):
    return random_distr(softmax(w,t))




weight_in_hid = np.random.rand(25, 256+field_size+1)/10
weight_hid_out = np.random.rand(1,25+1)/10

e_InHidden = np.zeros((25,256+field_size+1))
e_HiddenOut = np.zeros((1,25+1))




for epi in range(episodes):
    print 'Episode %i' % epi

    # Reset our Location
    current_location = 3

    # for every time in episode
    final_position = 6;
    for i in range(image_number):
        print '[' + str(current_location) + ']',
        image = images[:,i]

        board = np.zeros(field_size)
        board[current_location] = 1
        current_eval = evaulate_current( np.concatenate((image, board)), weight_in_hid, weight_hid_out)

        possible_locations = [0]
        if current_location > 0:
            possible_locations.append(-1)
        if current_location < field_size-1:
            possible_locations.append(1)

        outcomes = np.zeros(len(possible_locations))
        boards = np.zeros((len(possible_locations),field_size))

        for j in range(len(possible_locations)):
            location = possible_locations[j]
            boards[j,current_location+location] = 1

            outcomes[j] = evaulate_current( np.concatenate((image, boards[j,:])), weight_in_hid, weight_hid_out)

        print outcomes

        idx = picksoftmax(outcomes, 0.3)
        current_location += possible_locations[idx]
        next_board = boards[idx,:]

        next_eval = outcomes[idx]
        if i == (image_number-1):
            if current_location == final_position:
                next_eval = 3
            else:
                next_eval = 0

        weight_in_hid,weight_hid_out, e_InHidden, e_HiddenOut = back_prop(weight_in_hid,weight_hid_out, e_InHidden, e_HiddenOut, 0, current_eval, next_eval, np.concatenate((image,next_board)), alpha, ld)

    print 'END: [' + str(current_location) + ']',
    if current_location == 6:
        print 'GOOD',
        # current_location = 5
        # possible_locations = [0, -1, 1]
        # boards = np.zeros((len(possible_locations),field_size))
        # outcomes = np.zeros(len(possible_locations))
        # for j in range(3):
        #     location = possible_locations[j]
        #     boards[j,current_location+location] = 1
        #
        #     outcomes[j] = evaulate_current( np.concatenate((image, boards[j,:])), weight_in_hid, weight_hid_out)
        #
        # print outcomes
    print


scipy.io.savemat('output.mat', {'layer_1':weight_in_hid, 'layer_2':weight_hid_out, 'e_1': e_InHidden, 'e_2': e_HiddenOut})
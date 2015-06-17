__author__ = 'darioml'

# This file will allow for PNG images to be converted to a series of PNGs that relate to a specific speed for the ball.

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


matlab_file = scipy.io.loadmat('mat/ball.mat')

data = matlab_file['X']
labels = matlab_file['Y'][0]

# print len(np.where(labels == 0.)[0])
# print len(np.where(labels == 1.)[0])
# print len(np.where(labels == 2.)[0])
# print len(np.where(labels == 3.)[0])
#
# index = 1097+20
#
# print
# print
# print labels[index]
#
# plt.imshow( np.reshape(data[index,:], (16,16)), cmap=cm.Greys_r, interpolation="nearest")
# plt.show()
#
# print len(labels)

def show_image(data):
    fig = plt.figure()
    im1 = fig.add_subplot(121)
    im1.matshow(data[0:256].reshape((16,16)),cmap=cm.Greys_r)
    im1 = fig.add_subplot(122)
    im1.matshow(data[256:512].reshape((16,16)),cmap=cm.Greys_r)
    plt.show()


def create_image(full_data, index_1, index_2):
    return np.concatenate((full_data[index_1,:], data[index_2,:]))

def create_sequence(full_data, label, min, max, speed, amount):
    # speed is the number of movements the trajectory should be split into.
    # e.g. if there are 100 images, a speed of 20 means that two images are chosen with a distance of 5 images.
    #
    # Recommended to use speed of 3 for fast, 5 for medium speeds and 8 for slow trajectories.\
    # A suitable amount of images will be generated from that!

    no_imgs = int(max-min) + 1
    distance = no_imgs / speed

    no_potentials = no_imgs-distance

    idxs = np.linspace(min,min+no_potentials-1,amount, dtype=int)

    return np.concatenate((full_data[idxs,:],full_data[idxs+distance,:]),1), [label,speed]*np.ones((amount,2))


#sequences are hardcoded from knowledge

sequences = [[(0,17),(18,35),(36,52),(53,70),(71,86),(87,105),(106,123),(124,142)],
             [(144, 170), (171, 257), (258, 316), (317, 403), (404, 486), (487, 523)],
             [(525, 601), (602, 713), (714, 826), (827, 934), (935, 1030), (1031, 1096)],
             [(1097, 1117), (1117, 1138), (1138, 1161), (1161, 1184), (1184, 1205), (1205, 1227), (1227, 1250), (1250, 1275), (1275, 1297)]]

is_first = True
for speed in [3,5,8]:
    for sequence_set in sequences:
        for sequence in sequence_set:
            # print sequence
            if is_first is True:
                rtn,lab = create_sequence(data, labels[sequence[0]], sequence[0], sequence[1], 3, 10)
                is_first = False
            else:
                tmp,tml = create_sequence(data, labels[sequence[0]], sequence[0], sequence[1], speed, 10)
                rtn = np.concatenate((rtn, tmp),0)
                lab = np.concatenate((lab, tml))
                # print rtn.shape, lab.shape


# TESTING

# index = 700
#
# print lab[index]
# show_image(rtn[index,:])


scipy.io.savemat('mat/ball_with_speed.mat', {
    'X': rtn,
    'Y': lab
})


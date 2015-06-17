__author__ = 'darioml'

import scipy.io
import numpy as np

# There was a bug in the previous ball.mat, where random ball data was given a label by its self.
# This was wrong and this scrips aims to fix that!

matlab_file = scipy.io.loadmat('mat/ball_old.mat')

data = matlab_file['X']
labels = matlab_file['Y'][0]


print len(np.where(labels == 0.)[0])
print len(np.where(labels == 1.)[0])
print len(np.where(labels == 2.)[0])
print len(np.where(labels == 3.)[0])
print
print

new_data = np.delete(data,range(525,851),0)
new_labels = np.delete(labels,range(525,851),0)

print len(np.where(new_labels == 0.)[0])
print len(np.where(new_labels == 1.)[0])
print len(np.where(new_labels == 2.)[0])
print len(np.where(new_labels == 3.)[0])
print
print

scipy.io.savemat('mat/ball.mat', {'X': new_data, 'Y': new_labels})
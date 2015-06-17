from nn_simple import NN
import scipy.io

data = scipy.io.loadmat('../data/mat/ball.mat')

labels = data['Y'][0]
data   = data['X']
data = data/255


pat = [
    [data[5,:], [1,0,0,0]],
    [data[150,:], [0,1,0,0]],
    [data[550,:], [0,0,1,0]],
    [data[1100,:], [0,0,0,1]]
]


n = NN(256, 20, 4)

n.train(pat)
n.test(pat)
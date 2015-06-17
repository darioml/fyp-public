import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pprint import pprint
from basic.perception import Perception, Trainer

test = Perception(2)
train = Trainer(1.4,0)
train.train_2d(test, 50000)

fig, ax = plt.subplots()
orgline, = ax.plot([0, 0.5, 1], [0*train.grad + train.offset, 0.5*train.grad + train.offset, 1*train.grad + train.offset], color='k')
line, = ax.plot([], marker='x', color='r', ls='')
line2, = ax.plot([], marker='+', color='b', ls='')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


def update(data):
    # line.set_ydata(data)
    location = np.random.rand(2)
    ans      = test.feedforward(location)
    # real_ans = 1 if location[1] > (location[0] * train.grad + train.offset) else 0

    if ans == 1:
        line.set_xdata(np.append(line.get_xdata(), location[0]))
        line.set_ydata(np.append(line.get_ydata(), location[1]))
    else:
        line2.set_xdata(np.append(line2.get_xdata(), location[0]))
        line2.set_ydata(np.append(line2.get_ydata(), location[1]))

    ax.relim()
    ax.autoscale_view()

    return line,line2,


def data_gen():
    while True: yield np.random.rand(1)

ani = animation.FuncAnimation(fig, update, data_gen, interval=1)
plt.show()
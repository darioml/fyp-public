__author__ = 'darioml'


import numpy as np
import scipy.io
import agent as agent_helper
import agent.td_lambda, agent.td_zero
from environment import basic_maze


# SET UP OPTIONS
no_episodes = 10000
no_events_in_episode = 10
board_size = 5
num_dir = 4

# SET UP PARAMETERS and OBJECTS
alpha = .2
gamma = .8 #.8
ld    = .7
env = basic_maze.environment(board_size)
agent = agent.td_lambda.table(env, num_dir, alpha, gamma, ld)

# env.set_rewards([(0,4), (9,4), (4,9), (4,0)])
env.set_rewards([(0,2), (4,2), (2,4), (2,0)])

## Let's GO!

rewards = [];

for j in range(no_episodes):
    # reset location
    env.reset()

    # ball direction
    direction = np.random.randint(0,num_dir)

    for i in range(no_events_in_episode):
        x,y = env.current_location

        # Evaluate every possible move:
        possible_action = env.get_possible_actions()
        evaluations = agent.eval_actions(possible_action, direction, x, y)

        # Decide a location using the softmax
        action = agent.take_action(possible_action, evaluations)

        # tell the env what our action is
        is_episode_over = (i == no_events_in_episode-1)
        new_loc,reward = env.take_action(action, direction, is_episode_over)

        # update eligibility and state-value matrix based on reward
        agent.update_estimators(direction, x, y, reward, new_loc)

        # Let's get an indication of how good we are doing
        if (i == no_events_in_episode-1) is True:
            rewards.append(reward>0)
            print reward,
            if j%30 == 0:
                print

print
print

np.set_printoptions(linewidth=200)
print agent.V
scipy.io.savemat('../results/5_td_l_ld_7.mat', {'rw':rewards})
# scipy.io.savemat('../results/10_td_lambda.mat', {'V':agent.V})
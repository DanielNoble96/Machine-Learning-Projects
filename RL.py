import numpy as np
import random

rows = 6
cols = 25
num_tiles = rows * cols
dist = [0, 0, 0, 0, 1]
epsilon = 0.9
grid_rolledout = np.array([np.random.choice(dist) for i in range(num_tiles)])
grid = np.reshape(grid_rolledout, (rows, cols))
print(grid_rolledout)
print(grid)

position = [0, 0]
Q_table = np.zeros((150, 4))
print(Q_table)
reward = 0
time_reward = -1

while True:
    random_num = random.uniform(0, 1)
    if (random_num < 0.9):
        # explore
        action_idx = random.randrange(4)
        # check for bounds
        valid = isValid(action_idx, position)
        if not valid:
            while not valid:
                reward += time_reward
                action_idx = random.randrange(4)
                valid = isValid(action_idx, position)
 #       Q_table[
# loop until q table stops changing meaninfully or goal met
    # generate random number
    # if random number greater than epsilon, exploit
        # choose an action randomly
        # if not valid, (outside of bound),
            # loop until valid action
            # apply step penalty and do nothing
            # generate new action
            # end loop
        # new q value = current q + lr * (reward + gamma * max future q - current q)
        # update position
        # update reward (from step and state)
    # else explore
        # choose action randomly
        # if not valid, (outside of bound),
            # loop until valid action
            # apply step penalty and do nothing
            # generate new action
            # end loop
        # update position
        # update reward
    

    

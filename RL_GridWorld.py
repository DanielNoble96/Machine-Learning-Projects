import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, numRows, numCols, numActions):
        self.numRows = numRows
        self.numCols = numCols
        self.numStates = numRows * numCols
        self.numActions = numActions
        
        self.epsilon = 0.9
        self.alpha = 0.01
        self.gamma = 0.6
        self.path = []

        # s(i) = rowNum * 25 + colNum
        self.obstacle_dist = [0, 0, 0, 0, 1]
        self.grid_rolledout = np.arange(150)
 #       self.grid_rolledout = np.array([np.random.choice(obstacle_dist) for i in range(numStates)])
        self.grid = np.reshape(self.grid_rolledout, (numRows, numCols))
        self.position_2d = [2, 0]
        self.position_1d = 75


    def create_reward_matrix(self):
        # edit this for Obstacle and Litter module
        reward_matrix = np.zeros((self.numStates, self.numActions))

        for s in range(self.numStates):

            if ((s + 2) % 25 == 0):
                reward_matrix[s][2] = 100
                reward_matrix[s][0] = -5
            else:
                reward_matrix[s][2] = 5
                reward_matrix[s][0] = -5
                
        return reward_matrix

    def sidewalk_reward_matrix(self):
        reward_matrix = np.zeros((self.numStates, self.numActions))

        for s in range(self.numStates):
            row = s // 25

            # on the sidewalk
            if row < 4 and row > 2:
                if ((s + 2) % 25 == 0):
                    reward_matrix[s][0] = -5
                    reward_matrix[s][2] = 100
                    reward_matrix[s][1] = -10
                    reward_matrix[s][3] = -100
                else:
                    reward_matrix[s][2] = 10
                    reward_matrix[s][1] = -10
                    reward_matrix[s][0] = -5
                    reward_matrix[s][3] = -10
                                
            elif row == 4:
                reward_matrix[s][0] = -10
                reward_matrix[s][2] = -10
                reward_matrix[s][1] = 5
            elif row == 2:
                reward_matrix[s][0] = -10
                reward_matrix[s][2] = -10
                reward_matrix[s][3] = 5
            else:
                if ((s + 2) % 25 == 0):
                    reward_matrix[s][2] = 100
                    reward_matrix[s][0] = -5
                else:
                    reward_matrix[s][0] = -5
                    reward_matrix[s][2] = 5
                
        return reward_matrix
                

    def isValid(self, action_idx):
        valid = True
        # define left = 0, up = 1, right = 2, down = 3
        # left border and moving left
        if (action_idx == 0 and self.position_2d[1] == 0):
            valid = False
        # top border and moving up
        if (action_idx == 1 and self.position_2d[0] == 0):
            valid = False
        # right border and moving right
        if (action_idx == 2 and self.position_2d[1] == 24):
            valid = False
        # bottom border and moving down
        if (action_idx == 3 and self.position_2d[0] == 5):
            valid = False
        return valid


    def train(self):
        Q_table = np.zeros((self.numStates, self.numActions))
        R = self.create_reward_matrix()
        trail = []
        while True:
            trail = trail + self.position_2d
            # print(self.path)
            # self.path.append(self.position_2d)
            random_num = random.uniform(0, 1)
            if (random_num < 0.1):
            # EXPLORE
                action_idx = random.randrange(4)
                valid = self.isValid(action_idx)
                if (not valid):
                    while (not valid):
                        # TO DO: UPDATE REWARD
                        action_idx = random.randrange(4)
                        valid = self.isValid(action_idx)
            else:
                # EXPLOIT
                action_idx = np.argmax(R[self.position_1d])
                valid = self.isValid(action_idx)
                if (not valid):
                    while (not valid):
                        # TO DO: UPDATE REWARD
                        action_idx = random.randrange(4)
                        valid = self.isValid(action_idx)
            # find displacement
            if (action_idx == 0):
                self.position_2d[1] -= 1
                position_1d_prime = self.position_1d - 1
            elif (action_idx == 1):
                self.position_2d[0] -= 1
                position_1d_prime = self.position_1d - 25
            elif (action_idx == 2):
                self.position_2d[1] += 1
                position_1d_prime = self.position_1d + 1
            else:
                self.position_2d[0] += 1
                position_1d_prime = self.position_1d + 25

            # update Q table
 
            Q_table[self.position_1d, action_idx] = ((1 - self.alpha) * Q_table[self.position_1d, action_idx]
                                                     + self.alpha * (R[self.position_1d, action_idx]
                                                                     + self.gamma * np.amax(Q_table[position_1d_prime])))

            # update current state
            copy_prime = copy.deepcopy(position_1d_prime)
            self.position_1d = copy_prime
            # self.position_1d = position_1d_prime

            # you've reached the end
            if self.position_2d[1] == 24:
                trail = trail + self.position_2d
                break
            
        return trail


def main():
    world = GridWorld(6, 25, 4)
    path = world.train()
    print(path)
    x_path = []
    y_path = []
    counter = 0
    for i in path:
        if (counter % 2 == 0):
            y_path.append(i)
        else:
            x_path.append(i)
        counter += 1

    x_dots = []
    y_dots = []
    for i in range(25):
        for j in range(6):
            x_dots.append(i)
            y_dots.append(j)

    x_side = []
    y_side1 = []
    y_side2 = []

    x_obstacles = []
    y_obstacles = []
    

    plt.plot(x_dots, y_dots, '.')
 #   plt.plot(x_obstacles, y_obstacles, 'k-', label = 'agent path')
    plt.plot(x_path, y_path, 'k-', label = 'agent path')
 #   plt.plot(x_litter, y_litter, 'yo', label = 'litter')
 #   plt.plot(x_side, y_side1, 'b--', label = 'sidewalk')
 #   plt.plot(x_side, y_side2, 'b--')
    plt.plot(0, 2, 'go', label = 'start')
    plt.plot(24, 1, 'ro', label = 'end')
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    main()

import gym
import numpy as np
import random
from gym import error, spaces, utils
from gym.utils import seeding

class DaverEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.width = 10
        self.height = 10

        self.pos = np.array([0, 0])
        self.state = np.zeros((self.width,self.height))
        self.state[self.pos[0]][self.pos[1]] = 1
        self.state[4][4] = 2

        self.safe_location = np.array([4,4])

        self.action_space = spaces.Discrete(4)

    def step(self, action):
        if action == 0:
            if self.pos[1] > 0:
                self.pos[1] -= 1
        elif action == 1:
            if self.pos[1] < self.height - 1:
                self.pos[1] += 1
        elif action == 2:
            if self.pos[0] < self.width - 1:
                self.pos[0] += 1
        else:
            if self.pos[0] > 0:
                self.pos[0] -= 1
        
        s_ = self.pos

        if np.array_equal(s_, self.safe_location):
            reward = 20
            done = True
        else:
            reward = -1
            done = False 
        
        self.state = np.zeros((self.width,self.height))
        self.state[self.pos[0]][self.pos[1]] = 1
        self.state[4][4] = 2        


        encoded = self.pos[0] * 10 + self.pos[1]
        return encoded, reward, done

    
    def render(self):
        self.state = np.zeros((self.width,self.height))
        self.state[self.pos[0]][self.pos[1]] = 1
        self.state[4][4] = 2        
        print(self.state)



    def reset(self):
        self.state = np.zeros((self.width,self.height))
        random_pos = np.random.randint(10, size=2)
        self.state[random_pos[0]][random_pos[1]] = 1
        self.state[4][4] = 2 
        self.pos = random_pos


        return 0     



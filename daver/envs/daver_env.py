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
        self.action_space = spaces.Discrete(4)

        self.reset()

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
        self.enemy_move()
<<<<<<< HEAD
        if np.array_equal(s_, self.safe_location):
            reward = 5
            done = True
        # elif np.array_equal(s_, self.safe_location):
        #     reward = 1000
        #     done = True 
        else:
            reward = -1
=======
        if np.array_equal(s_, self.enemy_pos):
            reward = -1000
            done = True
        elif np.array_equal(s_, self.safe_location):
            reward = 10
            done = True 
        else:
            reward = 0
>>>>>>> 24b027d22b5d0930035d620d0dc9e7b3e672187e
            done = False
        
        self.state = np.zeros((self.width,self.height))
        self.state[self.pos[0]][self.pos[1]] = 1
        self.state[self.enemy_pos[0]][self.enemy_pos[1]] = 2
        
        encoded = self.pos[0] + self.pos[1] * 10 + self.enemy_pos[0] * 100 + self.enemy_pos[1] * 1000 
        return encoded, reward, done

    
    def render(self):
        self.state = np.zeros((self.width,self.height))
        self.state[self.safe_location[0]][self.safe_location[1]] = 3
        self.state[self.pos[0]][self.pos[1]] = 2
        self.state[self.enemy_pos[0]][self.enemy_pos[1]] = 1
        
        print(self.state)

    def enemy_move(self):
        if not np.array_equal(self.pos, self.enemy_pos):
<<<<<<< HEAD
            if np.random.uniform(0, 1) > 0.9:
                self.enemy_pos += np.random.randint(1, size=2)
            else:
                diff = self.enemy_pos - self.pos 
                if abs(diff[0]) > abs(diff[1]):
                    self.enemy_pos[0] -= diff[0] / abs(diff[0])
                else:
                    self.enemy_pos[1] -= diff[1] / abs(diff[1])



=======
            diff = self.enemy_pos - self.pos 
            if abs(diff[0]) > abs(diff[1]):
                self.enemy_pos[0] -= diff[0] / abs(diff[0])
            else:
                self.enemy_pos[1] -= diff[1] / abs(diff[1])




>>>>>>> 24b027d22b5d0930035d620d0dc9e7b3e672187e
    def get_frame(self):
        return self.state
        
    def reset(self):
        self.state = np.zeros((self.width,self.height))
        random_pos1 = np.random.randint(10, size=2)
        random_pos2 = np.random.randint(10, size=2)
        random_pos3 = np.random.randint(10, size=2)

        self.state[random_pos1[0]][random_pos1[1]] = 1
        self.state[random_pos2[0]][random_pos2[1]] = 2
        self.state[random_pos3[0]][random_pos3[1]] = 3
        self.enemy_pos = random_pos1
        self.pos = random_pos2
        self.safe_location = random_pos3


        return 0     



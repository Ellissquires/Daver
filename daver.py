import gym
import random
import daver

env = gym.make('daver-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(random.randint(0,3)) # take a random action
# env.close()
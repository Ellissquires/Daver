import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

import gym
import daver

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

environment = gym.envs.make("daver-v0")
environment.render()


class Daver:
    def __init__(self, environment, optimiser):
        self._state_size = 100
        self._action_size = environment.action_space.n
        self._optimiser = optimiser

        self.experience_replay = deque(maxlen=2000)

        self.gamma = 0.6
        self.epsilon = 0.1


        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimiser)
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights()) 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            x = environment.action_space.sample()
            # print(x)
            return x

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)


optimiser = Adam(learning_rate=0.01)
agent = Daver(environment, optimiser)

batch_size = 32
num_of_episodes = 10
timesteps_per_episode = 100
agent.q_network.summary()

for e in range(0, num_of_episodes):
    # Reset the enviroment
    state = environment.reset()
    state = state.flatten()
    # Initialize variables
    reward = 0
    terminated = False
    
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state)
        print(action)
        # Take action    
        next_state, reward, terminated, info = environment.step(action)
        print(next_state)
        next_state = next_state.flatten()
        agent.store(state, action, reward, next_state, terminated)
        state = next_state
        
        if terminated:
            agent.align_target_model()
            break
            
        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)
        
        if timestep%10 == 0:
            bar.update(timestep/10 + 1)
    
    bar.finish()
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        environment.render()
        print("**********************************")


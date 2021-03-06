import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar
import json

import gym
import daver

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

env = gym.envs.make("daver-v0")

q_table = np.zeros([10000, 4])

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
current_frame_dict = {}
pens = [0]

for i in range(1, 10001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        all_epochs.append(epochs)
        print(f"Episode: {i}")

print("Training finished.\n")
print(all_epochs)


# export = json.dumps(current_frame_dict)
# print(current_frame_dict)
# current_frame_dict = { k: v.tolist() for k, v in current_frame_dict.items() }



# export = json.dumps(current_frame_dict)
# print(current_frame_dict)
# current_frame_dict = { k: v.tolist() for k, v in current_frame_dict.items() }
with open('export.json', 'w') as json_file:
    json.dump(current_frame_dict, json_file, indent=4)


# export = json.dumps(current_frame_dict)
# print(current_frame_dict)
# current_frame_dict = { k: v.tolist() for k, v in current_frame_dict.items() }
with open('export.json', 'w') as json_file:
    json.dump(current_frame_dict, json_file, indent=4)

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 1000
max_move = 100
times_caught = 0
safe_deposited = 0
safe = []
for e in range(episodes):   
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    i = 0
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        # if reward == -1000:
        #     times_caught += 1
        if reward == 5:
            safe.append(e)
            safe_deposited += 1

        if (i == max_move):
            done = True

        i+=1
        env.render()
        if e in current_frame_dict:
            current_frame_dict[e].append(env.state.tolist())
        else:
            current_frame_dict[e] = [env.state.tolist()]
        
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

with open('export.json', 'w') as json_file:
    json.dump(current_frame_dict, json_file, indent=4)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
# print(f"Average penalties per episode: {total_penalties / episodes}")
print("Agent caught " + str(times_caught) + " times")
print("Agent deposited " + str(safe_deposited) + " times")
print(safe)

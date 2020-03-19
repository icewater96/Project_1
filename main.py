# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:39:13 2020

@author: JLLU
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

BUFFER_SIZE = int(1e5)  # size of replay buffer
BATCH_SIZE = 64   # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3    # for soft update of target parameters
LR = 5e-4   # learning rate
UPDATE_EVERY = 4   # how ofter to update the network

def randomly_move():
    # Only for demo the usage of env
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = np.random.randint(action_size)        # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))


class QNetwork(nn.Module):
    # State to action deep network
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        # Initialize a model
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        # Given state, compute probability for each action
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class RelayBuffer:
    # Fixed-size buffer to store experience tuples
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        # Randomly sample a batch of experience from memory
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)
    
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
    
    
class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Relay memory
        self.memory = RelayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in relay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough sample are available im relay memory, get randome sampel and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0):
        # Return actions for given state as per current policy
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_value = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy 
        if random.random() > eps:
            return np.argmax(action_value.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
                                 
    def learn(self, experiences, gamma):
        # Update parameters 
                                 
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q value (for next states) for target model
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_target_next * (1-dones))
                                
        Q_expected = self.qnetwork_local(states).gather(1, torch.LongTensor(actions.numpy()))
                                 
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
                                 
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                                 
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
                                     
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)    
            
            
#%% Train
def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
                
    agent = Agent(state_size, action_size, 1)
    
    scores = []
    scores_window = deque(maxlen=100)   # last 100 scores
    eps = eps_start
    max_window_score = 14.0
    mean_score_list = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            # state -> action
            action = int(agent.act(state, eps))
                    
            # take action, observe reward, get next_state
            env_info = env.step(action)[brain_name]
                
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = int(env_info.local_done[0])
            score += reward
     
            # agent populates ReleyBuffer and learn as needed
            agent.step(state, action, reward, next_state, done)
            
            # For the next t
            state = next_state
    
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        mean_score_list.append(np.mean(scores_window))
        
        if i_episode % 5 == 0:
            print(f'Episode {i_episode} average score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= max_window_score:
            print(f"A new window score has reached. Score = {np.mean(scores_window)} ")
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint window score {np.mean(scores_window)}.pth')
            max_window_score = np.mean(scores_window)
        if np.mean(scores_window) >= 0.5:    
            plt.figure()
            plt.plot(np.arange(1, len(mean_score_list)+1), mean_score_list)
            plt.xlabel('Episode')
            plt.ylabel('Mean Score')
            plt.box(True)
            plt.grid(True)
            plt.title('Mean score of last 100 episodes')
            plt.show()
            
            
if __name__ == '__main__':
    env = UnityEnvironment(file_name=r'data\Banana_Windows_x86_64\Banana_Windows_x86_64\Banana.exe')
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    
    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    train()
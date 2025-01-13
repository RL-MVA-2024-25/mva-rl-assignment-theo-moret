from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch 
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import random


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def act(self, observation, use_random=False):
        device = torch.device('cpu')
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        self.path = path + "/model_params.pt"
        torch.save(self.model.state_dict(), self.path)
        return None

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/model_params.pt"
        self.model = self.myModel(device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return None


    def get_config(self): #Parameters to be fixed, taken from lecture

        config = {
            'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 100000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.,
            'epsilon_decay_period': 15000,
            'epsilon_delay_decay': 100,
            'batch_size': 600,
            'gradient_steps': 2,
            'update_target_strategy': 'replace', 
            'update_target_freq': 400,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss()}
        
        return config 
    

    def myModel(self, device):

        stateDim, actionDim = env.observation_space.shape[0], env.action_space.n
        nbrNeurons = 256

        DQN = torch.nn.Sequential(nn.Linear(stateDim, nbrNeurons),
                          nn.ReLU(),
                          nn.Linear(nbrNeurons, nbrNeurons),
                          nn.ReLU(), 
                          nn.Linear(nbrNeurons, nbrNeurons),
                          nn.ReLU(), 
                          nn.Linear(nbrNeurons, nbrNeurons),
                          nn.ReLU(), 
                          nn.Linear(nbrNeurons, actionDim)).to(device)

        return DQN
    


    def greedy_action(network, model, state):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
    


    def gradient_step(self): #same as in Lecture
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 



    def train(self): #Inspiration from lecture

        # Get DQN network and target model 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.myModel(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        
        # Get config 
        config = self.get_config()

        ## Parameters
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)

        ## Greedy Part
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        ## DQN config
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        ## Target Network
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        
        # Get ReplayBuffer
        self.memory = ReplayBuffer(self.buffer_size, self.device)


        # Training loop

        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        max_episode = 200

        best_reward = 0
        test_freq = 1

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1

            if done or trunc:
                episode += 1

                #Check for improvments
                if episode % test_freq == 0:
                    reward = evaluate_HIV(self, nb_episode=1)
                    if reward > best_reward:
                        print(f'Reward: {reward} vs. Last Best Reward: {best_reward}')
                        best_reward = reward
                        self.best_model = deepcopy(self.model).to(self.device)
                        


                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)

        return episode_return

# Replay Buffer as in Lecture

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

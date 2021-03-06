import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import pandas as pd
from noise import OUNoise
from replay_buffer import BasicBuffer
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_dim = 30
action_dim = 1
hidden_size = 100
max_episodes = 60
batch_size = 200
gamma = 0.99
tau = 0.001
buffer_maxlen = 10000
actor_lr = 1e-3
critic_lr = 1e-3
DELTA_T = 0.1
action_high = 20
action_low = -20
REACTION_TIME = 10
path = './train40/'


class Environment:
    def __init__(self):
        # state = [v_f(t), Δv(t), Δs(t)]
        self.v_f = 0
        self.delta_v = 0
        self.delta_s = 0
        # next state = [v_f(t + 1), Δv(t + 1), Δs(t + 1)]
        self.v_f_next = 0
        self.delta_v_next = 0
        self.delta_s_next = 0
        self.reward = 0

    def step(self, a, v_l_next, s_f, v_f_obs, s_f_obs, state):
        # state = [v_f(t), Δv(t), Δs(t)]
        self.v_f = state[27]
        self.delta_v = state[28]
        self.delta_s = state[29]

        # v_f(t+1) = v_f(t) + a * Δt
        self.v_f_next = self.v_f + a * DELTA_T
        # Δv(t+1) = v_l(t+1) - v_f(t+1)
        self.delta_v_next = v_l_next - self.v_f_next
        # Δs(t+1) = Δs(t) + (Δv(t+1) + Δv(t)) * Δt / 2
        self.delta_s_next = self.delta_s + 0.5 * DELTA_T * (self.delta_v_next + self.delta_v)

        # get next state
        # next_state = [v_f_next, delta_v_next, delta_s_next]
        next_state = np.array([self.v_f_next, self.delta_v_next, self.delta_s_next], dtype=np.float32)
        next_state = np.concatenate([state[3:], next_state])

        # reward = log(|(s_f_obs - s_f_next) / s_f_obs|)
        self.reward_v = - np.log(np.abs(self.v_f_next / v_f_obs - 1) + 1e-8)

        return next_state, np.array(self.reward_v, dtype=np.float32)


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)
        self.tanh = nn.Tanh()
        self.max_action = action_high

        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear2.bias)

    def forward(self, obs):
        x = self.tanh(self.linear1(obs))
        x = self.tanh(self.linear2(x)) * self.max_action
        return x


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.la = nn.Linear(action_dim, obs_dim)
        self.linear1 = nn.Linear(obs_dim + obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

        nn.init.normal_(self.la.weight, mean=0, std=0.1)
        nn.init.uniform_(self.la.bias)
        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear2.bias)

    def forward(self, x, a):
        x = x.view([batch_size, -1])
        a = self.la(a)
        xa_cat = torch.cat([x, a], dim=1)
        xa = self.tanh(self.linear1(xa_cat))
        q = self.linear2(xa)

        return q


class DDPGAgent:

    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):

        self.env = env

        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau

        # initialize actor and critic networks
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=critic_learning_rate, alpha=0.9)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def load(self):
        self.actor.load_state_dict(torch.load('./trained_para/ddpgrt_actor.pth'))
        self.critic.load_state_dict(torch.load('./trained_para/ddpgrt_critic.pth'))
        self.actor_target.load_state_dict(torch.load('./trained_para/ddpgrt_actor_target.pth'))
        self.critic_target.load_state_dict(torch.load('./trained_para/ddpgrt_critic_target.pth'))
        print('Parameters Loaded')

    def get_action(self, obs):
        state = torch.FloatTensor(obs).to(device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()
        return action

    def update(self, batch_size):

        # sample random batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, _ = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)

        next_actions = self.actor_target.forward(next_state_batch)
        target_Q = self.critic_target(next_state_batch, next_actions)
        expected_Q = reward_batch + (self.gamma * target_Q).detach()

        curr_Q = self.critic.forward(state_batch, action_batch)

        # update critic
        critic_loss = F.mse_loss(curr_Q, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = - self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def calculate_rmspe(data, actor):
    # data = [v_f(t), Δv(t), Δs(t), v_l(t + 1), s_f(t), v_f(t + 1), s_f(t + 1)]
    v_f_next_set = []
    state = np.reshape(data[0, 0: 30], newshape=[-1])

    for k in range(len(data)):
        a = actor(torch.from_numpy(state).to(device))
        # v_f(t+1) = v_f(t) + a * Δt
        v_f_next = state[27] + a.item() * DELTA_T
        v_f_next_set.append(v_f_next)
        # Δv(t+1) = v_l(t+1) - v_f(t+1)
        delta_v_next = data[k, 30] - v_f_next
        # Δs(t+1) = Δs(t) + (Δv(t+1) + Δv(t)) * Δt / 2
        delta_s_next = state[29] + 0.5 * DELTA_T * (state[28] + delta_v_next)

        # update state
        next_state = np.array([v_f_next, delta_v_next, delta_s_next], dtype=np.float32)
        state = np.concatenate([state[3:], next_state])

    v_f_next_set = np.reshape(v_f_next_set, newshape=[-1, 1])
    v_f_obs = np.reshape(data[:, 32], newshape=[-1, 1])

    # calculate RMSPE
    numerator = np.sum(np.square(v_f_obs - v_f_next_set))
    denominator = np.sum(np.square(v_f_obs))
    RMSPEv = np.sqrt(numerator / denominator)
    return RMSPEv


# read data
dataset = np.zeros([30, 400, 34], dtype=np.float32)

for i in range(30):
    table_i = pd.read_csv(path + "train" + ((str)(i + 1)) + ".csv", sep=",", header=None, skiprows=1)
    train_data_i = getData(table_i)
    train_data_i = train_data_i[100: 500]
    dataset[i] = train_data_i

print('Reading data finished')

# train
env = Environment()
agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_learning_rate=critic_lr, actor_learning_rate=actor_lr)
agent.load()
noise = OUNoise()
counter = 0

for step in range(max_episodes):
    np.random.shuffle(dataset)
    for r in range(30):
        train_data = dataset[r]
        state = train_data[0, 0: obs_dim]

        for i in range(len(train_data)):
            counter = counter + 1
            if counter < 7000:
                action = np.random.normal(0, 1, size=action_dim)
                state = np.reshape(state, newshape=[obs_dim])
                next_state, reward = env.step(action.item(),
                                              train_data[i, 30],
                                              train_data[i, 31],
                                              train_data[i, 32],
                                              train_data[i, 33],
                                              state)
                agent.replay_buffer.push(state, action, reward, next_state, done=0)
                state = next_state
            else:
                state = np.reshape(state, newshape=[1, obs_dim])
                action = agent.get_action(state) + noise.sample()
                state = np.reshape(state, newshape=[obs_dim])
                next_state, reward = env.step(action.item(),
                                              train_data[i, 30],
                                              train_data[i, 31],
                                              train_data[i, 32],
                                              train_data[i, 33],
                                              state)
                agent.replay_buffer.push(state, action, reward, next_state, done=0)
                state = next_state
                if (i+1) % 200 == 0:
                    agent.update(batch_size)

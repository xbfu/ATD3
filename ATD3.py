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
        self.v_f = state[:, -1, 0]
        self.delta_v = state[:, -1, 1]
        self.delta_s = state[:, -1, 2]

        # v_f(t+1) = v_f(t) + a * Δt
        self.v_f_next = self.v_f + a * DELTA_T
        # Δv(t+1) = v_l(t+1) - v_f(t+1)
        self.delta_v_next = v_l_next - self.v_f_next
        # Δs(t+1) = Δs(t) + (Δv(t+1) + Δv(t)) * Δt / 2
        self.delta_s_next = self.delta_s + 0.5 * DELTA_T * (self.delta_v_next + self.delta_v)

        # get next state
        # next_state = [v_f_next, delta_v_next, delta_s_next]
        next_state = np.array([self.v_f_next, self.delta_v_next, self.delta_s_next], dtype=np.float32)
        next_state = np.reshape(next_state, newshape=[-1, 1, 3])
        next_state = np.concatenate([state[:, 1:, :], next_state], axis=1)

        # reward = log(|(s_f_obs - s_f_next) / s_f_obs|)
        self.reward_v = - np.log(np.abs(self.v_f_next / v_f_obs - 1) + 1e-8)

        return next_state, np.array(self.reward_v, dtype=np.float32)


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.wa1 = nn.Linear(hidden_size * 2, hidden_size)
        self.wa2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.wa1.weight)
        nn.init.uniform_(self.wa1.bias)
        nn.init.xavier_normal_(self.wa2.weight)
        nn.init.uniform_(self.wa2.bias)

    def forward(self, hidden, encoder_outputs):

        hidden = hidden.repeat([1, REACTION_TIME, 1])
        h_cat = torch.cat([hidden, encoder_outputs], 2)
        h_cat = self.tanh(self.wa1(h_cat))
        attn_energies = self.wa2(h_cat)

        return F.softmax(attn_energies, dim=1).permute(0, 2, 1)


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.rnn = nn.RNN(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.max_action = action_high

        nn.init.xavier_normal_(self.out.weight)
        nn.init.uniform_(self.out.bias)

    def forward(self, obs):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, hidden = self.rnn(obs, None)
        hidden = hidden.permute(1, 0, 2)
        energies = self.attn(hidden, r_out)
        context = energies.bmm(r_out)

        out = self.out(context.squeeze())
        x = self.tanh(out) * self.max_action
        return x, energies


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.tanh = nn.Tanh()

        # Q1
        self.la1 = nn.Linear(action_dim, obs_dim)
        self.linear1 = nn.Linear(obs_dim + obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

        nn.init.normal_(self.la1.weight, mean=0, std=0.1)
        nn.init.uniform_(self.la1.bias)
        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear2.bias)

        # Q2
        self.la2 = nn.Linear(action_dim, obs_dim)
        self.linear3 = nn.Linear(obs_dim + obs_dim, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.la2.weight, mean=0, std=0.1)
        nn.init.uniform_(self.la2.bias)
        nn.init.normal_(self.linear3.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear3.bias)
        nn.init.normal_(self.linear4.weight, mean=0, std=0.1)
        nn.init.uniform_(self.linear4.bias)

    def forward(self, x, a):
        x = x.view([batch_size, -1])

        a1 = self.la1(a)
        xa1 = torch.cat([x, a1], 1)
        q1 = self.tanh(self.linear1(xa1))
        q1 = self.linear2(q1)

        a2 = self.la1(a)
        xa2 = torch.cat([x, a2], 1)
        q2 = self.tanh(self.linear3(xa2))
        q2 = self.linear4(q2)

        return q1, q2

    def Q1(self, x, a):
        x = x.view([batch_size, -1])
        a1 = self.la1(a)
        xa1 = torch.cat([x, a1], 1)

        q1 = self.tanh(self.linear1(xa1))
        q1 = self.linear2(q1)

        return q1


class ATD3Agent:

    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):

        self.env = env

        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.counter = 0
        self.policy_freq = 2

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
        self.actor.load_state_dict(torch.load('./trained_para/atd3_actor.pth'))
        self.critic.load_state_dict(torch.load('./trained_para/atd3_critic.pth'))
        self.actor_target.load_state_dict(torch.load('./trained_para/atd3_actor_target.pth'))
        self.critic_target.load_state_dict(torch.load('./trained_para/atd3_critic_target.pth'))
        print('Parameters Loaded')

    def get_action(self, obs):
        state = torch.FloatTensor(obs).to(device)
        action, _ = self.actor(state)
        action = action.cpu().data.numpy().flatten()
        return action

    def update(self, batch_size):
        self.counter += 1

        # sample random batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, _ = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(device)
        state_batch = state_batch.view([batch_size, 10, 3])
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        next_state_batch = next_state_batch.view([batch_size, 10, 3])

        with torch.no_grad():

            # noise
            noise = (
                    torch.randn_like(action_batch) * self.policy_noise
            ).clamp_(min=-self.noise_clip, max=self.noise_clip)

            # a' = target_actor(s')
            next_actions, _ = self.actor_target.forward(next_state_batch)
            next_actions = (next_actions + noise).clamp(min=action_low, max=action_high)
            # target_Q1, target_Q2 = target_critic(s', a')
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_actions)
            # target_Q = min(target_Q1, target_Q2)
            target_Q = torch.min(target_Q1, target_Q2)
            # y_ = r + γ × Q_target
            target_Q = reward_batch + self.gamma * target_Q

        # curr_Q1, curr_Q2 = critic(s, a)
        curr_Q1, curr_Q2 = self.critic(state_batch, action_batch)

        # update critic
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.counter % self.policy_freq == 0:
            # update actor
            actions, _ = self.actor.forward(state_batch)
            actor_loss = - self.critic.Q1(state_batch, actions).mean()

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
    state = np.reshape(data[0, 0: 30], newshape=[1, 10, 3])

    for k in range(len(data)):
        a, energies = actor(torch.from_numpy(state).to(device))
        # v_f(t+1) = v_f(t) + a * Δt
        v_f_next = state[0, -1, 0] + a.item() * DELTA_T
        v_f_next_set.append(v_f_next)
        # Δv(t+1) = v_l(t+1) - v_f(t+1)
        delta_v_next = data[k, 30] - v_f_next
        # Δs(t+1) = Δs(t) + (Δv(t+1) + Δv(t)) * Δt / 2
        delta_s_next = state[0, -1, 2] + 0.5 * DELTA_T * (state[0, -1, 1] + delta_v_next)

        # update state
        next_state = np.array([v_f_next, delta_v_next, delta_s_next], dtype=np.float32)
        next_state = np.reshape(next_state, newshape=[1, 1, 3])
        state = np.concatenate([state[:, 1:, :], next_state], axis=1)

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
agent = ATD3Agent(env, gamma, tau, buffer_maxlen, critic_learning_rate=critic_lr, actor_learning_rate=actor_lr)
agent.load()
noise = OUNoise()
counter = 0

for step in range(max_episodes):
    np.random.shuffle(dataset)
    for r in range(30):
        train_data = dataset[r]
        state = np.reshape(train_data[0, 0: obs_dim], newshape=[-1, 10, 3])
        for i in range(len(train_data)):
            counter = counter + 1
            if counter < 7000:
                action = np.random.normal(0, 1, size=action_dim)
                next_state, reward = env.step(action.squeeze(),
                                              train_data[i, 30],
                                              train_data[i, 31],
                                              train_data[i, 32],
                                              train_data[i, 33],
                                              state)
                agent.replay_buffer.push(state, action, reward, next_state, done=0)
                state = next_state
            else:
                action = agent.get_action(state) + noise.sample()
                action = np.clip(a_min=-action_high, a_max=action_high, a=action)
                next_state, reward = env.step(action.squeeze(),
                                              train_data[i, 30],
                                              train_data[i, 31],
                                              train_data[i, 32],
                                              train_data[i, 33],
                                              state)
                agent.replay_buffer.push(state, action, reward, next_state, done=0)
                state = next_state
                if (i+1) % 200 == 0:
                    agent.update(batch_size)

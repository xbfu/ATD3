import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque
import pandas as pd

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
path = './train_data/'


def getData(table):
    """
    table -> data
    """
    # table = [s_f, v_f, s_l, v_l]
    # 0 - s_f: y of the FV at time t
    # 1 - v_f: velocity of the FV at time t
    # 2 - s_l: y of the LV at time t
    # 3 - v_l: velocity of the LV at time t
    table = np.array(table, dtype=np.float32)
    table_len = len(table)

    samples = []
    data = []

    for l in range(table_len - 2):
        v_f = table[l, 1]                                                       # v_f(t)
        v_f_obs = table[l + 1, 1]                                               # v_f(t + 1)
        delta_v = table[l, 3] - table[l, 1]                                     # Δv(t) = v_l(t) - v_f(t)
        delta_s = table[l, 2] - table[l, 0]                                     # Δs(t) = s_l(t) - s_f(t)
        v_l_next = table[l + 1, 3]                                              # v_l(t + 1)
        s_f = table[l, 0]                                                       # s_f(t)
        s_f_obs = table[l + 1, 0]                                               # s_f(t + 1)

        # sample = [v_f(t), Δv(t), Δs(t), v_l(t + 1), s_f(t), v_f(t + 1), s_f(t + 1)]
        sample = np.array([v_f, delta_v, delta_s, v_l_next, s_f, v_f_obs, s_f_obs])
        samples.append(sample)

    samples = np.array(samples, dtype=np.float32)
    samples = np.reshape(samples, newshape=[-1, 7])

    for i in range(table_len - 20):
        state = np.reshape(samples[i: i + 10, 0: 3], newshape=[1, -1])
        item = np.concatenate([state.squeeze(), samples[i + 9, 3: 7]])
        data.append(item)

    data = np.array(data, dtype=np.float32)

    return data


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
        next_state = np.array([self.v_f_next, self.delta_v_next, self.delta_s_next], dtype=np.float32)
        next_state = np.concatenate([state[3:], next_state])

        # reward = log(|(v_f_obs - v_f_next) / v_f_obs|)
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

        # Initialize Actor and Critic Networks
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=critic_learning_rate, alpha=0.9)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def load(self):
        self.actor.load_state_dict(torch.load('./trained_para/ddpgrt_actor.pth'))
        self.critic.load_state_dict(torch.load('./trained_para/ddpgrt_critic.pth'))
        self.actor_target.load_state_dict(torch.load('./trained_para/ddpgrt_actor_target.pth'))
        self.critic_target.load_state_dict(torch.load('./trained_para/ddpgrt_critic_target.pth'))
        print('Model Loaded ')

    def save(self):
        torch.save(self.actor.state_dict(), './trained_para/ddpgrt_actor.pth')
        torch.save(self.actor_target.state_dict(), './trained_para/ddpgrt_actor_target.pth')
        torch.save(self.critic.state_dict(), './trained_para/ddpgrt_critic.pth')
        torch.save(self.critic_target.state_dict(), './trained_para/ddpgrt_critic_target.pth')
        print('Model Saved')

    def get_action(self, obs):
        state = torch.FloatTensor(obs).to(device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()
        return action

    def update(self, batch_size, i):
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

        RMSPE = np.zeros([30, 1])
        RMSPE_test = np.zeros([10, 1])

        RMSPE[0] = calculate_rmspe(test_data1, self.actor)
        RMSPE[1] = calculate_rmspe(test_data2, self.actor)
        RMSPE[2] = calculate_rmspe(test_data3, self.actor)
        RMSPE[3] = calculate_rmspe(test_data4, self.actor)
        RMSPE[4] = calculate_rmspe(test_data5, self.actor)
        RMSPE[5] = calculate_rmspe(test_data6, self.actor)
        RMSPE[6] = calculate_rmspe(test_data7, self.actor)
        RMSPE[7] = calculate_rmspe(test_data8, self.actor)
        RMSPE[8] = calculate_rmspe(test_data9, self.actor)
        RMSPE[9] = calculate_rmspe(test_data10, self.actor)
        RMSPE[10] = calculate_rmspe(test_data11, self.actor)
        RMSPE[11] = calculate_rmspe(test_data12, self.actor)
        RMSPE[12] = calculate_rmspe(test_data13, self.actor)
        RMSPE[13] = calculate_rmspe(test_data14, self.actor)
        RMSPE[14] = calculate_rmspe(test_data15, self.actor)
        RMSPE[15] = calculate_rmspe(test_data16, self.actor)
        RMSPE[16] = calculate_rmspe(test_data17, self.actor)
        RMSPE[17] = calculate_rmspe(test_data18, self.actor)
        RMSPE[18] = calculate_rmspe(test_data19, self.actor)
        RMSPE[19] = calculate_rmspe(test_data20, self.actor)
        RMSPE[20] = calculate_rmspe(test_data21, self.actor)
        RMSPE[21] = calculate_rmspe(test_data22, self.actor)
        RMSPE[22] = calculate_rmspe(test_data23, self.actor)
        RMSPE[23] = calculate_rmspe(test_data24, self.actor)
        RMSPE[24] = calculate_rmspe(test_data25, self.actor)
        RMSPE[25] = calculate_rmspe(test_data26, self.actor)
        RMSPE[26] = calculate_rmspe(test_data27, self.actor)
        RMSPE[27] = calculate_rmspe(test_data28, self.actor)
        RMSPE[28] = calculate_rmspe(test_data29, self.actor)
        RMSPE[29] = calculate_rmspe(test_data30, self.actor)

        RMSPE_test[0] = calculate_rmspe(test_data31, self.actor)
        RMSPE_test[1] = calculate_rmspe(test_data32, self.actor)
        RMSPE_test[2] = calculate_rmspe(test_data33, self.actor)
        RMSPE_test[3] = calculate_rmspe(test_data34, self.actor)
        RMSPE_test[4] = calculate_rmspe(test_data35, self.actor)
        RMSPE_test[5] = calculate_rmspe(test_data36, self.actor)
        RMSPE_test[6] = calculate_rmspe(test_data37, self.actor)
        RMSPE_test[7] = calculate_rmspe(test_data38, self.actor)
        RMSPE_test[8] = calculate_rmspe(test_data39, self.actor)
        RMSPE_test[9] = calculate_rmspe(test_data40, self.actor)

        aver_train = np.mean(RMSPE)
        aver_test = np.mean(RMSPE_test)

        print('Epoch%3s' % (step + 1),
              '| NO.%5d' % (i + 1),
              '| Train RMSPE: %8.5f' % aver_train,
              '| Test RMSPE: %7.4f' % aver_test,
              '| Actor Loss: %7.2f' % actor_loss,
              '| Critic Loss: %5.2f ' % critic_loss,
              )

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return aver_train


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


# Ornstein-Ulhenbeck Noise
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size=1, seed=2, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


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
    # calculate rmspe
    numerator = np.sum(np.square(v_f_obs - v_f_next_set))
    denominator = np.sum(np.square(v_f_obs))
    RMSPE = np.sqrt(numerator / denominator)
    return RMSPE


# Read data
dataset = np.zeros([30, 400, 34], dtype=np.float32)
for i in range(30):
    table_i = pd.read_csv(path + "train" + ((str)(i + 1)) + ".csv", sep=",", header=None, skiprows=1)
    train_data_i = getData(table_i)
    train_data_i = train_data_i[100: 500]
    dataset[i] = train_data_i

table = pd.read_csv(path + "train1.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data1 = test_data[100: 500]

table = pd.read_csv(path + "train2.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data2 = test_data[100: 500]

table = pd.read_csv(path + "train3.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data3 = test_data[100: 500]

table = pd.read_csv(path + "train4.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data4 = test_data[100: 500]

table = pd.read_csv(path + "train5.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data5 = test_data[100: 500]

table = pd.read_csv(path + "train6.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data6 = test_data[100: 500]

table = pd.read_csv(path + "train7.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data7 = test_data[100: 500]

table = pd.read_csv(path + "train8.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data8 = test_data[100: 500]

table = pd.read_csv(path + "train9.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data9 = test_data[100: 500]

table = pd.read_csv(path + "train10.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data10 = test_data[100: 500]

table = pd.read_csv(path + "train11.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data11 = test_data[100: 500]

table = pd.read_csv(path + "train12.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data12 = test_data[100: 500]

table = pd.read_csv(path + "train13.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data13 = test_data[100: 500]

table = pd.read_csv(path + "train14.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data14 = test_data[100: 500]

table = pd.read_csv(path + "train15.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data15 = test_data[100: 500]

table = pd.read_csv(path + "train16.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data16 = test_data[100: 500]

table = pd.read_csv(path + "train17.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data17 = test_data[100: 500]

table = pd.read_csv(path + "train18.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data18 = test_data[100: 500]

table = pd.read_csv(path + "train19.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data19 = test_data[100: 500]

table = pd.read_csv(path + "train20.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data20 = test_data[100: 500]

table = pd.read_csv(path + "train21.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data21 = test_data[100: 500]

table = pd.read_csv(path + "train22.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data22 = test_data[100: 500]

table = pd.read_csv(path + "train23.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data23 = test_data[100: 500]

table = pd.read_csv(path + "train24.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data24 = test_data[100: 500]

table = pd.read_csv(path + "train25.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data25 = test_data[100: 500]

table = pd.read_csv(path + "train26.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data26 = test_data[100: 500]

table = pd.read_csv(path + "train27.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data27 = test_data[100: 500]

table = pd.read_csv(path + "train28.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data28 = test_data[100: 500]

table = pd.read_csv(path + "train29.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data29 = test_data[100: 500]

table = pd.read_csv(path + "train30.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data30 = test_data[100: 500]

table = pd.read_csv(path + "train31.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data31 = test_data[100: 500]

table = pd.read_csv(path + "train32.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data32 = test_data[100: 500]

table = pd.read_csv(path + "train33.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data33 = test_data[100: 500]

table = pd.read_csv(path + "train34.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data34 = test_data[100: 500]

table = pd.read_csv(path + "train35.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data35 = test_data[100: 500]

table = pd.read_csv(path + "train36.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data36 = test_data[100: 500]

table = pd.read_csv(path + "train37.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data37 = test_data[100: 500]

table = pd.read_csv(path + "train38.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data38 = test_data[100: 500]

table = pd.read_csv(path + "train39.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data39 = test_data[100: 500]

table = pd.read_csv(path + "train40.csv", sep=",", header=None, skiprows=1)
test_data = getData(table)
test_data40 = test_data[100: 500]

print('Reading data finished')

env = Environment()
agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_learning_rate=critic_lr, actor_learning_rate=actor_lr)
agent.load()
noise = OUNoise()
counter = 0
start_error = 0.5

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
                    aver_train = agent.update(batch_size, i)
                    if aver_train < start_error:
                        agent.save()
                        start_error = aver_train

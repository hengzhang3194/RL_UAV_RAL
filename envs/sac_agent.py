import os, uuid, random
import numpy as np

import torch
from torch.distributions import Normal

import gymnasium as gym


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity

        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.position = 0

    def push(self, obs, act, rew, next_obs, ter, tru):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, act, rew, next_obs, ter, tru)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, ter, tru = map(np.stack, zip(*batch))  # stack for each element
        return obs, act, rew, next_obs, ter, tru

    def pour(self):
        obs, act, rew, next_obs, ter, tru = map(np.stack, zip(*self.buffer))  # stack for each element
        return obs, act, rew, next_obs, ter, tru


class ContinuousActionValueNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_dims:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims + action_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], 1))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state, action):
        x = torch.concatenate([state, action], dim=1)
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class ContinuousPolicyNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_dims:int, hidden_size:list, log_std_range=(-20, 2)):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_dims))  # second to last layer (-2) mean of action
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_dims))  # last layer (-1) log_std of action
        self.layers = torch.nn.ModuleList(self.layers)

        self.log_std_range = log_std_range

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-2):
            x = torch.nn.functional.relu(self.layers[i](x))
        mean = self.layers[-2](x)
        log_std = torch.clamp(self.layers[-1](x), *self.log_std_range)
        return mean, log_std


class SacAgent(object):
    '''Soft Actor-Critic (SAC-v2) algorithm for continuous action space | https://arxiv.org/abs/1812.05905
    '''

    def __init__(self, obs_dims:int, act_dims:int, hidden_dims:list, gamma=0.99, tau=0.005, q_lr=3e-4, pi_lr=3e-4, a_lr=3e-4, auto_entropy=True, device:str='cpu', **kwargs):
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        self.device = torch.device(device)
        
        self.entropy_target = - act_dims

        self.q_1_net = ContinuousActionValueNetwork(obs_dims, act_dims, hidden_dims).to(self.device)
        self.q_1_optim = torch.optim.Adam(self.q_1_net.parameters(), lr=q_lr)

        self.q_2_net = ContinuousActionValueNetwork(obs_dims, act_dims, hidden_dims).to(self.device)
        self.q_2_optim = torch.optim.Adam(self.q_2_net.parameters(), lr=q_lr)

        self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_dims).to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi_net.parameters(), lr=pi_lr)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=self.device)  # FIXME set_train and save_model like 'torch.nn'
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=a_lr)

        # target network for 'over-value' problem
        self.q_1_net_ = ContinuousActionValueNetwork(obs_dims, act_dims, hidden_dims).to(self.device)
        for param, param_ in zip(self.q_1_net.parameters(), self.q_1_net_.parameters()):
            param_.data.copy_(param.data)

        self.q_2_net_ = ContinuousActionValueNetwork(obs_dims, act_dims, hidden_dims).to(self.device)
        for param, param_ in zip(self.q_2_net.parameters(), self.q_2_net_.parameters()):
            param_.data.copy_(param.data)

    @property
    def name(self):
        return 'SAC'

    def set_train(self):
        self.q_1_net.train()
        self.q_2_net.train()
        self.pi_net.train()

    def set_eval(self):
        self.q_1_net.eval()
        self.q_2_net.eval()
        self.pi_net.eval()

    def save_model(self, dir):
        if not os.path.exists(dir): 
            os.makedirs(dir)
        torch.save(self.q_1_net.state_dict(), os.path.join(dir, "q_1.pth"))
        torch.save(self.q_2_net.state_dict(), os.path.join(dir, "q_2.pth"))
        torch.save(self.pi_net.state_dict(), os.path.join(dir, "pi.pth"))

    def load_model(self, dir):
        assert os.path.exists(dir), f"Directory '{dir}' of weights and biases is NOT exist."
        self.q_1_net.load_state_dict(torch.load(os.path.join(dir, "q_1.pth")))
        self.q_2_net.load_state_dict(torch.load(os.path.join(dir, "q_2.pth")))
        self.pi_net.load_state_dict(torch.load(os.path.join(dir, "pi.pth")))

    def get_action(self, state:np.ndarray, deterministic=False) -> np.ndarray:  # only be called by outside
        mean, log_std = self.pi_net(torch.FloatTensor(state).to(self.device))
        std = log_std.exp()
        if deterministic:
            act = torch.tanh(mean)
        else:
            # 1. re-parameterization trick: (mean + std * N(0,1)), also can be replaced by using 'Normal(mean, std).rsample()'
            # 2. 'tanh()' is used to convert infinity value into finity value within (-1, 1)
            z = Normal(0, 1).sample(mean.shape).to(self.device)
            act = torch.tanh(mean + std * z)
        return act.detach().cpu().numpy()
    
    def _get_action(self, state:torch.Tensor):  # only be called by inside
        mean, log_std = self.pi_net(state)
        std = log_std.exp()

        ##############################################
        has_nan_mean = torch.isnan(mean).any().item()
        if has_nan_mean:
            print(f"find NaN value {mean}")
            mean = torch.nan_to_num(mean)

        has_nan_std = torch.isnan(std).any().item()
        if has_nan_std:
            print(f"find NaN value {std}")
            std = torch.nan_to_num(std)
        ##############################################
        
        z = Normal(0, 1).sample(mean.shape).to(self.device)
        u = mean + std * z
        act = torch.tanh(u)
        
        eps = torch.finfo(torch.float32).eps
        log_prob = Normal(mean, std).log_prob(u) - torch.log(1.0 - act.pow(2) + eps) - np.log(1.0) # 对log的概率函数进行修正
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return act, log_prob
    
    def update(self, batch_buff):
        self.set_train()

        obs, act, rew, next_obs, ter, tru = batch_buff

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).unsqueeze(1).to(self.device)
        ter = torch.FloatTensor(np.float32(ter)).unsqueeze(1).to(self.device)
        tru = torch.FloatTensor(np.float32(tru)).unsqueeze(1).to(self.device)

        # TODO reward re-scale

        # update action-value function (Q function) using TD (Temporal Difference) method
        next_act, next_log_prob = self._get_action(next_obs)
        next_q = torch.min(self.q_1_net_(next_obs, next_act), self.q_2_net_(next_obs, next_act)) - self.log_alpha.exp() * next_log_prob
        td_q = rew + (1 - ter) * self.gamma * next_q

        curr_q_1 = self.q_1_net(obs, act)  # use old_act cause the next_obs is induced by old_obs and old_act
        curr_q_2 = self.q_2_net(obs, act)
        q_1_loss = torch.nn.MSELoss()(curr_q_1, td_q.detach())
        q_2_loss = torch.nn.MSELoss()(curr_q_2, td_q.detach())

        self.q_1_optim.zero_grad()
        q_1_loss.backward()
        self.q_1_optim.step()
        
        self.q_2_optim.zero_grad()
        q_2_loss.backward()
        self.q_2_optim.step()  

        # update policy function (pi function)
        pred_act, pred_log_prob = self._get_action(obs)  # TODO use new 'act' is correct?
        pred_q_value = torch.min(self.q_1_net(obs, pred_act), self.q_2_net(obs, pred_act))
        pi_loss = (self.log_alpha.exp() * pred_log_prob - pred_q_value).mean()  # TODO why pred_q_value.detach() is wrong?

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        # update alpha (wrt entropy)
        if self.auto_entropy:  # auto entropy: trade-off between exploration (max entropy) and exploitation (max Q)
            _, pred_log_prob = self._get_action(obs)
            log_alpha_loss = (- self.log_alpha.exp() * (pred_log_prob + self.entropy_target).detach()).mean()
            
            self.log_alpha_optim.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optim.step()
        else:
            self.log_alpha = 0.0
            log_alpha_loss = 0.0

        # update the target action-value function (Target Q function) softly
        for param, param_ in zip(self.q_1_net.parameters(), self.q_1_net_.parameters()):
            param_.data.copy_(param.data * self.tau + param_.data * (1.0 - self.tau))
        for param, param_ in zip(self.q_2_net.parameters(), self.q_2_net_.parameters()):
            param_.data.copy_(param.data * self.tau + param_.data * (1.0 - self.tau))
        
        return {'loss/q_1':q_1_loss.detach().cpu().numpy(), 'loss/q_2':q_2_loss.detach().cpu().numpy(), 'loss/pi':pi_loss.detach().cpu().numpy(), 'loss/log_alpha':log_alpha_loss.detach().cpu().numpy()}
    
    def learn(self, env:gym.Env, buffer, logger, batch_size:int=256, max_steps:int=1e6, verbose:int=0):
        raise NotImplementedError


if __name__ == '__main__':
    # hyper-params
    max_steps = 1e6
    batch_size = 256
    average_range = 50

    # env
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2'][0]
    env = gym.make(env_name)

    # buffer
    buffer = ReplayBuffer(1e6)

    # agent
    assert isinstance(env.action_space, gym.spaces.Box), "Only support CONTINUOUS action space yet."
    agent = SacAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_dims=[256, 256], gamma=0.99, tau=0.005, q_lr=3e-4, pi_lr=3e-4, a_lr=3e-4, device='cuda')

    # logger
    from torch.utils.tensorboard import SummaryWriter
    uid = str(uuid.uuid1()).split('-')[0]
    logger = SummaryWriter(log_dir=f"./tensorboard/{env_name}/{agent.name}/{uid}")

    # training
    steps, episodes, returns = 0, 0, []
    while steps <= max_steps:
        episode_rew = 0
        episode_len = 0

        obs, _ = env.reset()
        while True:  # one rollout
            act = agent.get_action(obs, deterministic=False)
            next_obs, rew, ter, tru, _ = env.step(act)
                
            buffer.push(obs, act, rew, next_obs, ter, tru)
            obs = next_obs

            steps += 1
            episode_len += 1
            episode_rew += rew
            
            if len(buffer) > batch_size:
                loss_log = agent.update(buffer.sample(batch_size))
                for key, value in loss_log.items():
                    logger.add_scalar(key, value, steps)

            if ter or tru:
                break
        
        episodes += 1

        returns.append(episode_rew)
        average_return = np.array(returns).mean() if len(returns) <= average_range else np.array(returns[-(average_range+1):-1]).mean()

        # verbose
        print(f"UID: {uid} | Steps: {steps} | Episodes: {episodes} | Episode Length: {episode_len} | Episode Reward: {episode_rew} | Average Return: {average_return}")
        
        # logging
        logger.add_scalar('episodic/return', episode_rew, steps)
        logger.add_scalar('episodic/length', episode_len, steps)
        logger.add_scalar('episodic/return(average)', average_return, steps)
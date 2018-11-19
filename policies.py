import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

def get_policy(args, env):
    if args.environment == 'walker':
        if args.alg == 'ES':
            # add the model on top of the convolutional base
            policy = ESPolicyContinuous(24, 4)
        elif args.alg == 'PPO' or 'COMBO':
            policy = PPOPolicyContinuous(24, 4)
    elif args.environment == 'cartpole':
        if args.alg == 'ES':
            # add the model on top of the convolutional base
            policy = ESPolicyDiscrete(4, 2)
        elif args.alg == 'PPO' or 'COMBO':
            policy = PPOPolicyDiscrete(4, 2)
    if torch.cuda.is_available():
        policy = policy.cuda()
    return policy

class ESPolicyDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ESPolicyDiscrete, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Softmax())

    def forward(self, state, stochastic=True):
        pred = self.policy(state)
        if stochastic:
            dist = Categorical(pred)
            return dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            return pred.argmax() # depends on action space type (box or discrete)

class ESPolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ESPolicyContinuous, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Tanh())

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, stochastic=False):
        mean = self.policy(state)
        dist = Normal(mean, F.softplus(self.std))
        if stochastic:
            return dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            return mean.squeeze() # depends on action space type (box or discrete)

class PPOPolicyDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicyDiscrete, self).__init__()
        body = nn.Sequential(
        			nn.Linear(state_dim, 100),
        			nn.ReLU(),
        			nn.Linear(100, 100))

        self.policy = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, action_dim),
        			nn.Softmax(dim=1))

        self.vf = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, 1))

    def forward(self, state, stochastic=True):
        pred = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Categorical(pred)
        if stochastic:
            action = dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            action = pred.argmax() # depends on action space type (box or discrete)
        log_prob = dist.log_prob(action).squeeze()
        return value, action, log_prob

    def evaluate(self, state, action):
        pred = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Categorical(pred)
        log_prob = dist.log_prob(action).squeeze()
        entropy = dist.entropy().squeeze()
        return value, log_prob, entropy


class PPOPolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicyContinuous, self).__init__()
        body = nn.Sequential(
        			nn.Linear(state_dim, 100),
        			nn.ReLU(),
        			nn.Linear(100, 100))

        self.policy = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, action_dim),
        			nn.Tanh())

        self.vf = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, 1))
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, stochastic=True):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        if stochastic:
            action = dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            action = mean.squeeze() # depends on action space type (box or discrete)
        log_prob = dist.log_prob(action).sum(-1).squeeze()
        return value, action, log_prob

    def evaluate(self, state, action):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        log_prob = dist.log_prob(action).sum(dim=1).squeeze()
        entropy = dist.entropy().sum(dim=1).squeeze()
        return value, log_prob, entropy

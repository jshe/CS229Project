import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def get_policy(args):
    if args.alg == 'ES':
        # add the model on top of the convolutional base
        policy = ESPolicy()
    elif args.alg == 'PPO':
        policy = PPOPolicy()
    if torch.cuda.is_available():
        policy = policy.cuda()
    return policy

class ESPolicy(nn.Module):
    def __init__(self):#, obs_space, action_space):
        super(ESPolicy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax())

    def forward(self, state, stochastic=True):
        pred = self.policy(state)
        if stochastic:
            dist = Categorical(pred)
            return dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            return pred.argmax() # depends on action space type (box or discrete)

class PPOPolicy(nn.Module):
    def __init__(self):#, obs_space, action_space):
        super(PPOPolicy, self).__init__()
        body = nn.Sequential(
        			nn.Linear(4, 100),
        			nn.ReLU(),
        			nn.Linear(100, 100))

        self.policy = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, 2),
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

    # def evaluate_state(self, state):
    #     return self.vf(state)

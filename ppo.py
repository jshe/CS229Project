import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time
import utils
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPOModule:

    def __init__(
        self,
        policy,
        env_func,
        n_updates=5,
        batch_size=64,
        max_steps=256,
        gamma=0.99,
        clip=0.01,
        ent_coeff=0.0,
        learning_rate=0.0001
    ):
        np.random.seed(int(time.time()))
        self.policy = policy
        self.weights = list(self.policy.parameters())
        self.env_function = env_func
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.clip = clip
        self.ent_coeff = ent_coeff
        self.learning_rate = learning_rate
        # self.decay = decay

        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def step(self):
        # s_t, a_t, b(s_t) = v(s_t), \pi_{\theta_{\text{old}}}(a_t|s_t), R_t(\tau)
        states, actions, rewards, values, logprobs, returns = self.env_function(policy=self.policy, max_steps=self.max_steps, gamma=self.gamma)
        # \hat{A_t}(\tau) = R_t(\tau) - b(s_t)
        advantages = returns - values

        advantages = (advantages - advantages.mean()) / advantages.std()
        for update in range(self.n_updates):
            sampler = BatchSampler(SubsetRandomSampler(list(range(advantages.shape[0]))), batch_size=self.batch_size, drop_last=False)
            for i, index in enumerate(sampler):
                sampled_states = utils.to_var(states[index])
                sampled_actions = utils.to_var(actions[index])
                sampled_logprobs = utils.to_var(logprobs[index])
                sampled_returns = utils.to_var(returns[index])
                sampled_advs = utils.to_var(advantages[index])
                # v(s_t), \pi_\theta(a_t|s_t), H(\pi(a_t, |a_t))
                new_values, new_logprobs, dist_entropy = self.policy.evaluate(sampled_states, sampled_actions)

                # \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
                ratio1 = torch.exp(new_logprobs - sampled_logprobs)
                # [\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}]_{\text{clip}}
                ratio2 = ratio1.clamp(1-self.clip, 1+self.clip)
                # \min\{.,[.]_{\text{clip}}\}
                ratio = torch.min(ratio1, ratio2)
                # \min\{. \,[.]_{\text{clip}}\}
                policy_loss = -sampled_advs.detach() * ratio
                sampled_returns = sampled_returns.view(-1, 1)
                new_values = new_values.view(-1, 1)
                # \frac{1}{2}(v(s_t) - R_t(\tau))^2
                value_loss = F.mse_loss(new_values, sampled_returns)
                loss = policy_loss.mean() + value_loss.mean() - self.ent_coeff * dist_entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.weights)

    @property
    def model_name(self):
        return "PPO_{}_{}_{}_{}_{}_{}_{}".format(
                self.n_updates,
                self.batch_size,
                self.max_steps,
                self.gamma,
                self.clip,
                self.ent_coeff,
                self.learning_rate)

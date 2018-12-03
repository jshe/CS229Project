import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time
import utils
from multiprocessing.pool import ThreadPool
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class MaxPPOModule:

    def __init__(
        self,
        policy,
        env_func,
        population_size=50,
        sigma=0.1,
        n_updates=5,
        batch_size=64,
        max_steps=256,
        gamma=0.99,
        clip=0.01,
        ent_coeff=0.0,
        ppo_learning_rate=0.0001,
        threadcount=4
    ):
        self.policy = policy
        self.weights = list(self.policy.parameters())
        self.env_function = env_func
        self.population_size = population_size
        self.sigma = sigma
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.clip = clip
        self.ent_coeff = ent_coeff
        self.ppo_learning_rate = ppo_learning_rate
        # self.decay = decay
        self.pool = ThreadPool(threadcount)
        self.criterion = nn.MSELoss()

    def perturb_weights(self, weights, epsilons=[]):
        new_weights = []
        for i, weight in enumerate(weights):
            # \sigma*\epsilon
            perturb = utils.to_var(self.sigma * epsilons[i])
            # \Theta = \bar{\theta} + \sigma*\epsilon
            new_weights.append(weight.data + perturb)
        return new_weights

    def step(self):
        epsilons_population = []
        for _ in range(self.population_size):
            epsilons = []
            for weight in self.weights:
                # \epsilon_i \sim \mathcal{N}(0, I)
                epsilons.append(np.random.randn(*weight.data.size()))
            epsilons_population.append(epsilons)
        # R(\tau_i; \Theta)
        results = self.pool.map(
           self.ppo_step,
           [self.perturb_weights(copy.deepcopy(self.weights), epsilons=eps) for eps in epsilons_population]
        )
        rewards = [result[0] for result in results]
        print(rewards)
        ind = np.argmax(rewards)
        new_weights_population = [result[1] for result in results]
        for index, weight in enumerate(self.weights):
            new_weights = np.array([new_weights[index] for new_weights in new_weights_population])
            weight.data = new_weights[ind]
        return copy.deepcopy(self.weights)

    def ppo_step(self, weights):
        cloned_policy = copy.deepcopy(self.policy)
        for i, weight in enumerate(cloned_policy.parameters()):
            try:
                weight.data.copy_(weights[i])
            except:
                weight.data.copy_(weights[i].data)
        optimizer = optim.Adam(cloned_policy.parameters(), lr=self.ppo_learning_rate)

        for _ in range(10):
            # s_t, a_t, b(s_t) = v(s_t), \pi_{\theta_{\text{old}}}(a_t|s_t), R_t(\tau)
            states, actions, rewards, values, logprobs, returns = self.env_function(cloned_policy, max_steps=self.max_steps, gamma=self.gamma)#, stochastic=False)
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
                    new_values, new_logprobs, dist_entropy = cloned_policy.evaluate(sampled_states, sampled_actions)

                    ratio = torch.exp(new_logprobs - sampled_logprobs)
                    # print(ratio.sum())
                    sampled_advs = sampled_advs.view(-1, 1)
                    surrogate1 = ratio * sampled_advs
                    surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * sampled_advs
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    # # \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
                    # ratio1 = torch.exp(new_logprobs - sampled_logprobs)
                    # # [\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}]_{\text{clip}}
                    # ratio2 = ratio1.clamp(1-self.clip, 1+self.clip)
                    # # \min\{.,[.]_{\text{clip}}\}
                    # ratio = torch.min(ratio1, ratio2)
                    # # \min\{. \,[.]_{\text{clip}}\}
                    # policy_loss = -sampled_advs.detach() * ratio
                    sampled_returns = sampled_returns.view(-1, 1)
                    new_values = new_values.view(-1, 1)
                    # \frac{1}{2}(v(s_t) - R_t(\tau))^2
                    value_loss = F.mse_loss(new_values, sampled_returns)
                    loss = policy_loss.mean() + value_loss.mean() - self.ent_coeff * dist_entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        rewards = self.env_function(cloned_policy, stochastic=False, render=False, reward_only=True)
        new_weights = list(cloned_policy.parameters())
        return rewards, new_weights

    @property
    def model_name(self):
        return "COMBO_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                self.population_size,
                self.sigma,
                self.n_updates,
                self.batch_size,
                self.max_steps,
                self.gamma,
                self.clip,
                self.ent_coeff,
                self.ppo_learning_rate)

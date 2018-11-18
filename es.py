import copy
import pickle
import time
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
import utils

class ESModule:
    def __init__(
        self,
        policy,
        env_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        # decay=1.0,
        # sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        reward_goal=None,
        consecutive_goal_max=None,
        save_path=None
    ):
        np.random.seed(int(time.time()))
        self.weights = list(policy.parameters())
        self.env_function = env_func
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        # self.decay = decay
        # self.sigma_decay = sigma_decay
        self.pool = ThreadPool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_max = consecutive_goal_max
        self.consecutive_goal_count = 0
        self.save_path = save_path

    def perturb_weights(self, weights, epsilons=[]):
        new_weights = []
        for i, weight in enumerate(weights):
            # \sigma*\epsilon
            perturbed = utils.to_var(self.sigma * epsilons[i])
            # \Theta = \bar{\theta} + \sigma*\epsilon
            new_weights.append(weight.data + perturbed)
        return new_weights

    def run(self, iterations, print_step=10):
        #for iteration in range(iterations):
        iteration = 0
        while True:
            epsilons_population = []
            for _ in range(self.population_size):
                epsilons = []
                for weight in self.weights:
		            # \epsilon_i \sim \mathcal{N}(0, I)
                    epsilons.append(np.random.randn(*weight.data.size()))
                epsilons_population.append(epsilons)
            # R(\tau_i; \Theta)
            rewards = self.pool.map(
                self.env_function,
                [self.perturb_weights(copy.deepcopy(self.weights), epsilons=eps) for eps in epsilons_population]
            )

            if np.std(rewards) != 0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                for index, weight in enumerate(self.weights):
                    epsilons = np.array([epsilons[index] for epsilons in epsilons_population])
                    # sum_{i=1}^k \epsilon_i R(\tau_i) (7)
                    rewards_population = utils.to_var(np.dot(epsilons.T, normalized_rewards).T)
                    # \bar{\theta} = \bar{theta} + \dfrac{1}{k\sigma} sum_{i=1}^k \epsilon_i R(\tau_i)a (7)
                    weight.data = weight.data + self.learning_rate / (self.population_size * self.sigma) * rewards_population
                    # self.learning_rate *= self.decay
                    # self.sigma *= self.sigma_decay

            if (iteration+1) % print_step == 0:
                weights = copy.deepcopy(self.weights)
                test_reward = self.env_function(weights, render=self.render_test)
                print('iter %d. reward: %f' % (iteration+1, test_reward))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))

                if self.consecutive_goal_max and self.reward_goal:
                    self.consecutive_goal_count = self.consecutive_goal_count+1 if test_reward >= self.reward_goal else 0
                    if self.consecutive_goal_count >= self.consecutive_goal_max:
                        return self.weights
            iteration += 1
        return self.weights

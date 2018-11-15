import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time
from torch.distributions.categorical import Categorical

# from evostra import EvolutionStrategy
from es import ESModule
from ppo import PPOModule
# from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str, required=True, help='Path to save final weights')
    parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
    parser.add_argument('-a', '--alg', type=str, help='ES or PPO')

    parser.set_defaults(cuda=False)
    args = parser.parse_args()
    return args

def get_reward_PPO(cuda, model, vf, render=False, stochastic=True):
    env = gym.make("CartPole-v0")
    state = env.reset()
    done = False
    total_reward = 0
    traj = []
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        batch = torch.from_numpy(state[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = model(Variable(batch))
        if stochastic:
            action = Categorical(prediction).sample().numpy()
        else:
            action = prediction.argmax(dim=1).numpy()
        state, reward, done, _ = env.step(action[0])
        value = vf(torch.Tensor(state).unsqueeze(0)).squeeze().detach().numpy()
        # value = value[0]
        traj.append((state, action, reward, value))
        total_reward += reward

    env.close()
    return total_reward, traj

def get_reward_ES(weights, cuda, model, render=False, stochastic=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("CartPole-v0")
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        batch = torch.from_numpy(state[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch))
        if stochastic:
            action = Categorical(prediction).sample().numpy()
        else:
            action = prediction.data.numpy().argmax()
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward

def get_model(args, cuda):
    if args.alg == 'ES':
        # add the model on top of the convolutional base
        model = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax()
        )
        # model.apply(weights_init)
        if cuda:
            model = model.cuda()
        return model

    elif args.alg == 'PPO':
        body = nn.Sequential(
        			nn.Linear(4, 100),
        			nn.ReLU(),
        			nn.Linear(100, 100)
        			)

        policy = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, 2),
        			nn.Softmax(dim=1)
        			)

        vf = nn.Sequential(
        			body,
        			nn.ReLU(),
        			nn.Linear(100, 1))
        if cuda:
            policy, vf = policy.cuda(), vf.cuda()
        return (policy, vf)

def main():
    args = parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    model = get_model(args, cuda)
    gym_logger.setLevel(logging.CRITICAL)
    if args.alg == 'ES':
        mother_parameters = list(model.parameters())
        partial_func = partial(get_reward_ES, cuda=cuda, model=model)
        alg = EvolutionModule(
            mother_parameters, partial_func, population_size=5, sigma=0.1,
            learning_rate=0.001, threadcount=15, cuda=cuda, reward_goal=200,
            consecutive_goal_stopping=10
        )
    elif args.alg == 'PPO':
        partial_func = partial(get_reward_PPO, cuda=cuda, model=model[0], vf=model[1])
        alg = PPOModule(
            model, partial_func, n_episodes=5, cuda=cuda, reward_goal=200,
            consecutive_goal_stopping=10
        )
    start = time.time()
    final_weights = alg.run(50)
    end = time.time() - start

    pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

    if args.alg == 'ES':
        reward = partial_func(final_weights, render=True)
    elif args.alg == 'PPO':
        reward, _ = partial_func(render=True)
    print(f"Reward from final weights: {reward}")
    print(f"Time to completion: {end}")

if __name__ == '__main__':
    main()

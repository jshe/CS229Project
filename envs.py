import numpy as np
import copy
import gym
import torch
import time
import utils
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

# weights is first argument because of threadpool
def run_env_ES(weights, policy, env_func, render=False, stochastic=False):
    cloned_policy = copy.deepcopy(policy)
    for i, weight in enumerate(cloned_policy.parameters()):
        try:
            weight.data.copy_(weights[i])
        except:
            weight.data.copy_(weights[i].data)
    env = env_func()
    state = env.reset()
    done = False
    total_reward = 0
    # TODO: do all enviroments have DONE?
    step  = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        action = cloned_policy.forward(utils.to_var(state).unsqueeze(0), stochastic) # depends on observation space (generally box)
        action = utils.to_data(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
    env.close()
    return total_reward

def run_env_PPO(policy, env_func, max_steps=100, render=False, stochastic=True, reward_only=False, gamma=0.99):
    env = env_func()
    state = env.reset()
    done = False
    total_reward = 0
    states = []
    actions = []
    rewards = []
    values = []
    logprobs = []
    masks = []
    step = 0
    while True:
        if (not reward_only) and (step == max_steps):
            break
        if render:
            env.render()
            time.sleep(0.05)
        value, action, logprob = policy.forward(utils.to_var(state).unsqueeze(0), stochastic) # depends on observation space (generally box)
        value, action, logprob = utils.to_data(value), utils.to_data(action), utils.to_data(logprob)
        # TODO: epsilon-greedy??
        state, reward, done, _ = env.step(action)
        # calculate value of the NEXT state
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        logprobs.append(logprob)
        masks.append(1-done)
        total_reward += reward
        if done:
            if reward_only:
                env.close()
                return total_reward
            else:
                state = env.reset()
        step += 1
    env.close()
    states = np.asarray(states)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    logprobs = np.asarray(logprobs)
    masks = np.asarray(masks)
    if done:
        last_value = 0.0
    else:
        last_value, _, _ = policy.forward(utils.to_var(state).unsqueeze(0), stochastic)
        last_value = utils.to_data(last_value)
    returns = calculate_returns(rewards, masks, last_value, gamma)
    return states, actions, rewards, values, logprobs, returns

def calculate_returns(rewards, masks, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    for i in reversed(range(rewards.shape[0])):
        # accumulate future if not done
        returns[i] = gamma * returns[i+1] * masks[i] + rewards[i]
    returns = np.asarray(returns[:-1])
    return returns

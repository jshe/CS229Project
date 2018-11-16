from functools import partial
import logging
import os
import pickle
import time
import torch
import models
import options
import rewards
from gym import logger as gym_logger
from es import ESModule
from ppo import PPOModule
# from pytorch_es.utils.helpers import weights_init

def main():
    args = options.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    model = models.get_model(args, cuda)
    gym_logger.setLevel(logging.CRITICAL)
    if args.alg == 'ES':
        reward_func = partial(rewards.get_reward_ES,
                              cuda=cuda,
                              model=model)
        alg = ESModule(
            model, reward_func,
            population_size=5, # HYPERPARAMETER
            sigma=0.1, # HYPERPARAMETER
            learning_rate=0.001, # HYPERPARAMETER
            threadcount=15,
            cuda=cuda,
            reward_goal=200,
            consecutive_goal_stopping=10
        )
    elif args.alg == 'PPO':
        reward_func = partial(rewards.get_reward_PPO,
                              cuda=cuda,
                              model=model[0],
                              vf=model[1])
        alg = PPOModule(
            model,
            reward_func,
            n_episodes=5, # HYPERPARAMETER
            cuda=cuda,
            reward_goal=200,
            consecutive_goal_stopping=10
        )
    start = time.time()
    final_weights = alg.run(50)
    end = time.time() - start
    pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

    if args.alg == 'ES':
        reward = reward_func(final_weights, render=True)
    elif args.alg == 'PPO':
        reward, _ = reward_func(render=True)
    print(f"Reward from final weights: {reward}")
    print(f"Time to completion: {end}")

if __name__ == '__main__':
    main()

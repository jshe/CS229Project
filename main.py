from functools import partial
import logging
import os
import pickle
import time
import torch
import policies
import options
import gym
import envs
from gym import logger as gym_logger
from es import ESModule
from ppo import PPOModule
# from pytorch_es.utils.helpers import weights_init

def main():
    args = options.parse_args()
    policy = policies.get_policy(args)
    gym_logger.setLevel(logging.CRITICAL)
    env_func = partial(envs.get_env, args=args)
    if args.alg == 'ES':
        run_func = partial(envs.run_env_ES,
                              policy=policy,
                              env_func=env_func)
        alg = ESModule(
            policy, run_func,
            population_size=5, # HYPERPARAMETER
            sigma=0.1, # HYPERPARAMETER
            learning_rate=0.001, # HYPERPARAMETER
            threadcount=15,
            reward_goal=200,
            consecutive_goal_max=10
        )
    elif args.alg == 'PPO':
        run_func = partial(envs.run_env_PPO,
                              policy=policy,
                              env_func=env_func,
                              max_steps=500) # TODO: update
        alg = PPOModule(
            policy,
            run_func,
            n_updates=5, # HYPERPARAMETER
            batch_size=64, # HYPERPARAMETER
            reward_goal=200,
            consecutive_goal_max=10)
    start = time.time()
    final_weights = alg.run(50)
    end = time.time() - start
    pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

    if args.alg == 'ES':
        total_reward = run_func(final_weights, render=False)
    elif args.alg == 'PPO':
        total_reward = run_func(render=False, reward_only=True)
    print(f"Reward from final weights: {total_reward}")
    print(f"Time to completion: {end}")

if __name__ == '__main__':
    main()

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
import numpy as np
from gym import logger as gym_logger
from es import ESModule
from ppo import PPOModule
import matplotlib.pyplot as plt
# from pytorch_es.utils.helpers import weights_init

def main():
    args = options.parse_args()
    gym_logger.setLevel(logging.CRITICAL)
    env_func = partial(envs.get_env, args=args)
    reward_goal = 200
    consecutive_goal_max = 10
    max_iteration = 5000
    all_rewards = []
    all_times = []
    all_totals = []
    for trial in range(args.n_trials):
        policy = policies.get_policy(args)
        if args.alg == 'ES':
            run_func = partial(envs.run_env_ES,
                                  policy=policy,
                                  env_func=env_func)
            alg = ESModule(
                policy, run_func,
                population_size=args.population_size, # HYPERPARAMETER
                sigma=args.sigma, # HYPERPARAMETER
                learning_rate=args.lr, # HYPERPARAMETER
                threadcount=4
            )
        elif args.alg == 'PPO':
            run_func = partial(envs.run_env_PPO,
                                  policy=policy,
                                  env_func=env_func,
                                  max_steps=args.max_steps) # TODO: update
            alg = PPOModule(
                policy,
                run_func,
                n_updates=args.n_updates, # HYPERPARAMETER
                batch_size=args.batch_size, # HYPERPARAMETER
                gamma=args.gamma,
                clip=args.clip,
                ent_coeff=args.ent_coeff,
                learning_rate=args.lr)
        exp_dir = os.path.join(args.directory, alg.model_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        start = time.time()
        consecutive_goal_count = 0
        iteration = 0
        rewards = []
        while True:
            if iteration >= max_iteration:
                break
            weights = alg.step()
            if (iteration+1) % 10 == 0:
                if args.alg == 'ES':
                    test_reward = run_func(weights, stochastic=False, render=False)
                elif args.alg == 'PPO':
                    test_reward = run_func(stochastic=False, render=False, reward_only=True)
                rewards.append(test_reward)
                print('iter %d. reward: %f' % (iteration+1, test_reward))

                if consecutive_goal_max and reward_goal:
                    consecutive_goal_count = consecutive_goal_count+1 if test_reward >= reward_goal else 0
                    if consecutive_goal_count >= consecutive_goal_max:
                        break
            iteration += 1
        end = time.time() - start
        if args.alg == 'ES':
            total_reward = run_func(weights, stochastic=False, render=False)
        elif args.alg == 'PPO':
            total_reward = run_func(stochastic=False, render=False, reward_only=True)
        all_rewards.append(rewards)
        all_times.append(end)
        all_totals.append(total_reward)
        print(f"Reward from final weights: {total_reward}")
        print(f"Time to completion: {end}")
    max_len = 0
    for rewards in all_rewards:
        if len(rewards) > max_len:
            max_len = len(rewards)
    for rewards in all_rewards:
        while len(rewards) < max_len:
            rewards.append(reward_goal)
        rewards = np.array(rewards)
    all_rewards = np.array(all_rewards)
    rewards_mean = np.mean(all_rewards, axis=0)
    rewards_std = np.std(all_rewards, axis=0)
    total_mean = np.mean(all_totals)
    time_mean = np.mean(all_times)
    plt.errorbar(np.arange(max_len), rewards_mean, yerr=rewards_std, label='rewards')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(exp_dir, "rewards_plot.png")
    plt.savefig(path)
    plt.close()
    pickle.dump(weights, open(os.path.join(exp_dir, 'weights.pkl'), 'wb'))
    out_file = open(os.path.join(exp_dir, "results.txt"), 'w')
    print(f"Average rewards from final weights: {total_mean}")
    msg = f"Average rewards from final weights: {total_mean}"
    msg += "\n"
    print(f"Average to completion: {time_mean}")
    msg += f"Average to completion: {time_mean}"
    msg += "\n"

    out_file.write(msg)
    out_file.flush()

if __name__ == '__main__':
    main()

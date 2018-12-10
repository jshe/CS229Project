import matplotlib
matplotlib.use('Agg')
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
from es_ppo import ESPPOModule
from max_ppo import MaxPPOModule
from alt_ppo import AltPPOModule
import matplotlib.pyplot as plt
import random

# from pytorch_es.utils.helpers import weights_init


def get_env(args):
    if args.environment == 'cartpole':
        env = gym.make("CartPole-v0")
    elif args.environment == 'walker':
        env = gym.make("BipedalWalker-v2")
    env.seed(np.random.randint(args.seed))
    return env


def get_goal(args):
    if args.environment == 'cartpole':
        return 200
    elif args.environment == 'walker':
        return 300

def main():
    args = options.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gym_logger.setLevel(logging.CRITICAL)
    env_func = partial(get_env, args=args)
    env = get_env(args)
    reward_goal = get_goal(args)
    consecutive_goal_max = 10
    max_iteration = args.epoch
    all_rewards = []
    all_times = []
    all_totals = []
    for trial in range(args.n_trials):
        policy = policies.get_policy(args, env)
        if args.alg == 'ES':
            run_func = partial(envs.run_env_ES,
                               policy=policy,
                               env_func=env_func)
            alg = ESModule(
                policy, run_func,
                population_size=args.population_size, # HYPERPARAMETER
                sigma=args.sigma, # HYPERPARAMETER
                learning_rate=args.lr, # HYPERPARAMETER TODO:CHANGE
                threadcount=args.population_size)
        
        elif args.alg == 'PPO':
            run_func = partial(envs.run_env_PPO,
                               env_func=env_func) # TODO: update
            alg = PPOModule(
                policy,
                run_func,
                n_updates=args.n_updates, # HYPERPARAMETER
                batch_size=args.batch_size, # HYPERPARAMETER
                max_steps = args.max_steps,
                gamma=args.gamma,
                clip=args.clip,
                ent_coeff=args.ent_coeff,
                learning_rate=args.lr) # TODO: CHANGE

        elif args.alg == 'ESPPO':
            run_func = partial(envs.run_env_PPO,
                               env_func=env_func)

            alg = ESPPOModule(
                policy,
                run_func,
                population_size=args.population_size, # HYPERPARAMETER
                sigma=args.sigma, # HYPERPARAMETER
                n_updates=args.n_updates, # HYPERPARAMETER
                batch_size=args.batch_size, # HYPERPARAMETER
                max_steps = args.max_steps,
                gamma=args.gamma,
                clip=args.clip,
                ent_coeff=args.ent_coeff,
                n_seq=args.n_seq,
                ppo_learning_rate=args.ppo_lr,
                es_learning_rate=args.es_lr,
                threadcount=args.population_size)

        elif args.alg == 'MAXPPO':
            run_func = partial(envs.run_env_PPO,
                               env_func=env_func)

            alg = MaxPPOModule(
                policy,
                run_func,
                population_size=args.population_size, # HYPERPARAMETER
                sigma=args.sigma, # HYPERPARAMETER
                n_updates=args.n_updates, # HYPERPARAMETER
                batch_size=args.batch_size, # HYPERPARAMETER
                max_steps = args.max_steps,
                gamma=args.gamma,
                clip=args.clip,
                ent_coeff=args.ent_coeff,
                n_seq=args.n_seq,
                ppo_learning_rate=args.ppo_lr,
                threadcount=args.population_size)
  

        elif args.alg == 'ALTPPO':
            run_func = partial(envs.run_env_PPO,
                               env_func=env_func)

            alg = AltPPOModule(
                policy,
                run_func,
                population_size=args.population_size, # HYPERPARAMETER
                sigma=args.sigma, # HYPERPARAMETER
                n_updates=args.n_updates, # HYPERPARAMETER
                batch_size=args.batch_size, # HYPERPARAMETER
                max_steps = args.max_steps,
                gamma=args.gamma,
                clip=args.clip,
                ent_coeff=args.ent_coeff,
                n_alt=args.n_alt,
                es_learning_rate=args.es_lr,
                ppo_learning_rate=args.ppo_lr,
                threadcount=args.population_size)


        if args.render:
            with open(os.path.join(args.directory, 'weights.pkl'),'rb') as fp:
                weights = pickle.load(fp)
                policy.load_state_dict(weights)

            if args.alg == 'ES':
                total_reward = run_func(weights, stochastic=False, render=True)
            else:
                total_reward = run_func(policy, stochastic=False, render=True, reward_only=True)
            print(f"Total rewards from episode: {total_rewards}")
            return


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
                else:
                    test_reward = run_func(policy, stochastic=False, render=False, reward_only=True)
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
        else:
            total_reward = run_func(policy, stochastic=False, render=False, reward_only=True)
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
    np.savetxt(os.path.join(exp_dir, 'rewards.txt'), rewards_mean)
    pickle.dump(weights, open(os.path.join(exp_dir, 'weights.pkl'), 'wb'))
    out_file = open(os.path.join(exp_dir, "results.txt"), 'w')
    print(f"Average rewards from final weights: {total_mean}")
    msg = f"Average rewards from final weights: {total_mean}"
    msg += "\n"
    print(f"Average time to completion: {time_mean}")
    msg += f"Average ime to completion: {time_mean}"
    msg += "\n"
    print(f"Results saved at: {exp_dir}")
    out_file.write(msg)
    out_file.flush()

if __name__ == '__main__':
    main()

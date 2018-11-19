import argparse

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./checkpoints', help='experiment directory')
    parser.add_argument('-a', '--alg', type=str, help='ES or PPO')

    parser.add_argument('--population_size', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=0.1)

    parser.add_argument('--max_steps', type=int, default=256)
    parser.add_argument('--n_updates', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--ent_coeff', type=float, default=0.0)
    parser.add_argument('--environment', type=str, default='cartpole', choices=['cartpole', 'walker'])

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_trials', type=int, default=5)

    args = parser.parse_args()
    return args

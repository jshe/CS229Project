import matplotlib as mpl
mpl.use('Agg')
import time, os, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    desc = "Rank the most highly scored images by the discriminator."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file_1', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--file_2', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--file_3', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--file_4', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--file_5', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--save_dir', type=str, default="", help='The location of the saved GAN.')

    return parser.parse_args()
    

def main():
    args = parse_args()
    if args is None:
        exit()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    rewards_1 = np.loadtxt(args.file_1, dtype=float)
    rewards_2 = np.loadtxt(args.file_2, dtype=float)
    rewards_3 = np.loadtxt(args.file_3, dtype=float)
    rewards_4 = np.loadtxt(args.file_4, dtype=float)
    rewards_5 = np.loadtxt(args.file_5, dtype=float)

    print(rewards_1)
    print(rewards_2)
    plt.figure(figsize=(10, 6))


    col = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    line1, = plt.plot(np.arange(len(rewards_1)), rewards_1, col[1], label='trial 1')
    line2, = plt.plot(np.arange(len(rewards_2)), rewards_2, col[2], label='trial 2')
    line2, = plt.plot(np.arange(len(rewards_3)), rewards_3, col[3], label='trial 3')
    line2, = plt.plot(np.arange(len(rewards_4)), rewards_4, col[4], label='trial 4')
    line2, = plt.plot(np.arange(len(rewards_5)), rewards_5, col[5], label='trial 5')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "rewards_plot.png")
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    main()

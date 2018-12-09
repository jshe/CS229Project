import matplotlib as mpl
mpl.use('Agg')
import torch
import time, os, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    desc = "Rank the most highly scored images by the discriminator."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file_1', type=str, default="", help='The location of the saved GAN.')
    parser.add_argument('--file_2', type=str, default="", help='The location of the saved GAN.')
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
    print(rewards_1)
    print(rewards_2)
    plt.figure(figsize=(10, 6))

    line1, = plt.plot(np.arange(len(rewards_1)), rewards_1, 'k.-', label='trial 1')
    line2, = plt.plot(np.arange(len(rewards_2)), rewards_2, 'r--', label='trial 2')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(args.save_dir, "rewards_plot.png")
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    main()

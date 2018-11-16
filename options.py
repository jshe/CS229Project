import argparse

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str, required=True, help='Path to save final weights')
    parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
    parser.add_argument('-a', '--alg', type=str, help='ES or PPO')
    parser.set_defaults(cuda=False)
    args = parser.parse_args()
    return args

import torch.nn as nn

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

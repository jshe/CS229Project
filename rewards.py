import numpy as np
import copy
import gym
import torch
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

# weights is first argument because of threadpool
def get_reward_ES(weights, cuda, model, render=False, stochastic=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("CartPole-v0")
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        batch = torch.from_numpy(state[np.newaxis,...]).float() # depends on observation space (generally space)
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch)) # depends on observation space (generally space)
        if stochastic:
            action = Categorical(prediction).sample().numpy() # depends on action space type (box or discrete)
        else:
            action = prediction.data.numpy().argmax() # depends on action space type (box or discrete)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward

    def get_reward_PPO(cuda, model, vf, render=False, stochastic=True):
        env = gym.make("CartPole-v0")
        state = env.reset()
        done = False
        total_reward = 0
        traj = []
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            batch = torch.from_numpy(state[np.newaxis,...]).float() # depends on observation space (generally space)
            if cuda:
                batch = batch.cuda() # depends on observation space (generally space)
            prediction = model(Variable(batch))
            # TODO: epsilon-greedy??
            if stochastic:
                action = Categorical(prediction).sample().numpy() # depends on action space type (box or discrete)
            else:
                action = prediction.argmax(dim=1).numpy() # depends on action space type (box or discrete)
            state, reward, done, _ = env.step(action[0])
            value = vf(torch.Tensor(state).unsqueeze(0)).squeeze().detach().numpy()
            traj.append((state, action, reward, value))
            total_reward += reward

        env.close()
        return total_reward, traj

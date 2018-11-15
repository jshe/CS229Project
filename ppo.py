import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym

class PPOModule:

    def __init__(
        self,
        model,
        reward_func,
        n_episodes=50,
        # sigma=0.1,
        # learning_rate=0.001,
        # decay=1.0,
        # sigma_decay=1.0,
        # threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None
    ):
        self.policy = model[0]
        self.vf = model[1]
        self.weights = list(self.policy.parameters())
        self.reward_function = reward_func
        self.n_episodes = n_episodes
        # self.SIGMA = sigma
        # self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        # self.decay = decay
        # self.sigma_decay = sigma_decay
        # self.pool = ThreadPool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path
        self.gamma = 0.9
        self.clip = 0.1

        self.optim_p = optim.RMSprop(self.policy.parameters(), lr=1e-4)
        self.optim_vf = optim.RMSprop(self.vf.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def run(self, iterations, print_step=10):
        for iteration in range(iterations):
            V = []
            V_ = []
            Ratio = []
            Ent = []
            for _ in range(self.n_episodes):
                rewards, traj = self.reward_function()

                batch_size = len(traj)-1
                s_batch = torch.zeros(batch_size, 4)
                a_batch = torch.zeros(batch_size, 1)
                v_batch_ = torch.zeros(batch_size, 1)

                v = traj[-1][-1]
                for i, tr in enumerate(reversed(traj[:-1])):
                    tr_list = list(tr)
                    v = tr[2] + self.gamma * v
                    tr_list.append(v)
                    traj[-2-i] = tuple(tr_list)

                    s_batch[-1-i] = torch.Tensor(traj[-2-i][0])
                    a_batch[-1-i] = torch.Tensor(traj[-2-i][1])
                    v_batch_[-1-i] = torch.Tensor([traj[-2-i][4]]).unsqueeze(0)
                v_batch = self.vf(s_batch)
                A_batch = v_batch_ - v_batch
                prob_batch = self.policy(s_batch)
                ent_batch = -(prob_batch * torch.log(prob_batch)).sum(dim=1)
                p_batch = prob_batch.gather(1, a_batch.long())
                ratio1 = p_batch/p_batch.detach()
                ratio2 = ratio1.clamp(1-self.clip, 1+self.clip)
                ratio = torch.min(ratio1, ratio2)

                Ratio.append(ratio)
                V.append(v_batch)
                V_.append(v_batch_)
                Ent.append(ent_batch)

            Ratio = torch.cat(Ratio, dim=0)
            V = torch.cat(V, dim=0)
            V_ = torch.cat(V_, dim=0)
            Ent = torch.cat(Ent, dim=0)

            A = V_ - V
            A = (A - A.mean()) / A.std()
            loss = -A.detach() * Ratio
            loss_v = 0.5*((V_-V).clamp(-self.clip, self.clip))**2
            loss = loss.mean() + 0.1 * loss_v.mean() + 0.01 * Ent.mean()
            self.optim_p.zero_grad()
            self.optim_vf.zero_grad()
            loss.backward()
            self.optim_p.step()
            self.optim_vf.step()

            if (iteration+1) % print_step == 0:
              test_reward, _ = self.reward_function(stochastic=False, render=self.render_test)
              print('iter %d. reward: %f' % (iteration+1, test_reward))
              if self.save_path:
                  pickle.dump(self.weights, open(self.save_path, 'wb'))

              if self.reward_goal and self.consecutive_goal_stopping:
                  if test_reward >= self.reward_goal:
                      self.consecutive_goal_count += 1
                  else:
                      self.consecutive_goal_count = 0

                  if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                      return self.weights

        return self.weights

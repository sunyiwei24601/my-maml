import torch
import torch.nn.functional as F


class Episode(object):
    def __init__(self, gamma=0.99):
        self._rewards = []
        self._actions = []
        self._observations = []
        self._log_probs = []
        self._rtg = []  # reward to go
        self.gamma = gamma
        self._advantages = []
        pass

    @property
    def log_probs(self):
        """
        convert log_prob into a tensor, but this will clear the grad
        :return: 
        """
        return torch.as_tensor(self._log_probs)

    @property
    def length(self):
        return len(self._observations)

    @property
    def advantages(self):
        return torch.as_tensor(self._advantages)


    @property
    def rtg(self):
        return torch.as_tensor(self._rtg)

    @property
    def rewards(self):
        return torch.as_tensor(self._rewards)

    @property
    def actions(self):
        return torch.stack(self._actions)

    @property
    def observations(self):
        return torch.as_tensor(self._observations)

    def add(self, observation, action, reward, log_prob):
        self._rewards.append(reward)
        self._actions.append(action)
        self._observations.append(observation)
        self._log_probs.append(log_prob)

    def compute_rtg(self):
        self._rtg = [0 for _ in self._rewards]
        reward = 0
        for i in range(len(self._rewards) - 1, -1, -1):
            reward = self._rewards[i] + self.gamma * reward
            self._rtg[i] = reward

    def get_train_loss(self, policy=None, params=None):
        pi = policy(self.observations, params=params)
        log_probs = pi.log_prob(self.actions)
        # to keep the grad and grad_fn, we should calculate them separately
        loss = self.advantages * log_probs

        return sum(loss) / len(loss)

    def compute_advantages(self, baseline, gae_lambda=1.0):
        """
        Calculate the Advantages of this episode, which can be used to get loss function
        The formula here we use can be found in https://arxiv.org/abs/1506.02438 In subsection 3 Equation 15, 16
        WE USE GAE(Generalized advantage estimator in exponentially-weighted average
        if lambda = 0, gae = r_t + gamma * V(s_{t+1}) - V(s_t)
        if lambda = 1, gae = Reward_to_go(from s_t) - V(s_t)
        like the difference from 1 step to k step

        :param baseline: the value function model
        :param gae_lambda: exponentially-weight parameter lambda in GAE
        :return:
        """

        values = baseline(self.observations).detach().squeeze(1)
        # extend the values list, so we can calculate delta, (0,1) means extend last dimension 0 left, 0 right
        # default pad with constant 0
        values = F.pad(values, (0, 1))

        advantages = torch.zeros_like(self.rewards)
        rewards = torch.as_tensor(self._rewards)
        deltas = self.rewards + self.gamma * values[1:] - values[:-1]

        gae = torch.zeros(1, dtype=torch.float32)
        for i in range(len(self._rewards) - 1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i]
            advantages[i] = gae
        self._advantages = advantages
        # TODO Do we need Normalization here ?

        return advantages

import torch
from episode import Episode
from torch.optim import Adam
import numpy as np

class TaskWorker(object):
    def __init__(self, policy, env, baseline, train_episodes=None, test_episodes=None, inner_loop_steps=1, policy_lr=1e-3, max_env_steps=200):
        self.env = env
        self.max_env_steps = max_env_steps
        self.policy = policy
        self.policy_lr = policy_lr
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.inner_loop_steps = inner_loop_steps
        self.baseline = baseline
        pass

    def run(self):
        params = None
        # the step times of training parameter update
        train_episode_pairs = []
        for i in range(self.inner_loop_steps):
            train_episode = self.create_episode(params)
            train_episode_pairs.append(train_episode)
            loss = train_episode.get_train_loss(self.policy, params=params)
            params = self.policy.update_params(loss)

        self.train_episodes.append(train_episode_pairs)

        test_episode = self.create_episode(params)
        self.test_episodes.append(test_episode)
        return train_episode_pairs, test_episode

    def create_episode(self, params, gae_lambda=1.0, gamma=0.99):
        episode = Episode(gamma)
        for observation, action, reward, log_prob in self.create_trajectories(params):
            episode.add(observation, action, reward, log_prob.detach())
        episode.compute_rtg()

        self.baseline.fit(episode)
        episode.compute_advantages(self.baseline, gae_lambda=gae_lambda)
        return episode

    def create_trajectories(self, params=None):
        observation = self.env.reset()
        done = False
        step = 0
        # set no_grad to save memory. Creating trajectories will not involve backward() or gradient
        # but we calculate log_prob here, so we still need grad
        # with torch.no_grad()
        while step < self.max_env_steps:
            observation_tensor = torch.from_numpy(observation)
            pi = self.policy(observation_tensor, params)
            action_tensor = pi.sample()
            action = action_tensor.cpu().numpy()
            action[action==float("-inf")] = -1
            action[action==float("inf")] = 1

            new_observation, reward, done, _ = self.env.step(action)
            yield observation, action_tensor, reward, pi.log_prob(action_tensor).sum(dim=-1)
            observation = new_observation
            step += 1

import torch
from episode import Episode
from torch.optim import Adam

class TaskWorker(object):
    def __init__(self, policy, env, policy_lr=1e-3):
        self.env = env
        self.policy = policy
        self.policy_lr = policy_lr
        pass

    def run(self):
        train_episode = Episode()
        for observation, action, reward, log_prob in self.create_trajectories():
            train_episode.add(observation, action, reward, log_prob)
        optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        loss = train_episode.get_train_loss()
        optimizer.zero_grad()
        loss.backward()



    def create_trajectories(self):
        observation = self.env.reset()
        done = False
        with torch.no_grad():
            while not done:
                observation_tensor = torch.from_numpy(observation)
                pi = self.policy(observation_tensor)
                action_tensor = pi.sample()
                action = action_tensor.cpu().numpy()

                new_observation, reward, done, _ = self.env.step(action)
                yield observation, action_tensor, reward, pi.log_prob(action_tensor).sum(dim=-1)

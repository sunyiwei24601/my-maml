class Episode(object):
    def __init__(self, gamma=0.99):
        self.rewards = []
        self.actions = []
        self.observations = []
        self.log_probs = []
        self.rtg = []
        self.gamma = gamma
        pass

    def add(self, observation, action, reward, log_prob):
        self.rewards.append(reward)
        self.actions.append(action)
        self.observations.append(observation)
        self.log_probs.append(log_prob)

    def get_train_loss(self):
        self.rtg = [0 for _ in self.rewards]
        reward = 0
        for i in range(len(self.rewards), -1, -1):
            reward = self.rewards[i] + self.gamma * reward
            self.rtg[i] = reward
        loss = 0
        for i in range(self.rtg):
            loss += self.rtg[i] + self.log_probs[i]

        return loss


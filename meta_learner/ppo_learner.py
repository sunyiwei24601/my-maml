from meta_learner.base_learner import BaseMetaLearner, detach_distribution
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class PPOMetaLearner(BaseMetaLearner):
    def __init__(self, policy, fast_lr=0.5, first_order=False, device="cpu"):
        super(PPOMetaLearner, self).__init__(policy, device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    def adapt(self, train_episodes, first_order=None):
        """
            update train episodes' params, return updated params
        """
        if first_order is None:
            first_order = self.first_order
        # use new policy params, update the params by previous train trajectories if the advantage of this (s, 
        # a) is higher, please increase the probilities to choose action a when facing state s 
        params = None
        for train_episode in train_episodes:
            inner_loss = train_episode.get_train_loss(self.policy, params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def surrogate_loss(self, train_episodes, test_episode, epsilon=5e-2, old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        params = self.adapt(train_episodes, first_order=first_order)
        pi = self.policy(test_episode.observations, params=params)

        if old_pi is None:
            old_pi = detach_distribution(pi)
        log_ratio = (pi.log_prob(test_episode.actions) - old_pi.log_prob(test_episode.actions))
        ratio = torch.exp(log_ratio)
        clamp_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * test_episode.advantages
        losses = - torch.min(ratio * test_episode.advantages, clamp_ratio)

        return losses.mean(), old_pi

    def step(self, train_episodes, test_episodes, epsilon=5e-2, policy_iters=50, policy_learning_rate=1e-3):
        num_tasks = len(train_episodes)
        surrogate_losses = [self.surrogate_loss(train_episode, test_episode, epsilon)
                            for (train_episode, test_episode) in zip(train_episodes, test_episodes)
                            ]
        old_losses = [_[0] for _ in surrogate_losses]
        old_pis = [_[1] for _ in surrogate_losses]

        old_loss = sum(old_losses) / num_tasks
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_learning_rate)
        loss = old_loss
        for i in range(policy_iters):
            surrogate_losses = [self.surrogate_loss(train_episode, test_episode, epsilon, old_pi)
                                for (train_episode, test_episode, old_pi) in zip(train_episodes, test_episodes, old_pis)
                                ]
            losses = [_[0] for _ in surrogate_losses]

            loss = sum(losses) / num_tasks
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return loss
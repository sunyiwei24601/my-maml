import torch
from torch.distributions import Independent, Categorical, Normal


def detach_distribution(pi):
    if isinstance(pi, Independent):
        distribution = Independent(detach_distribution(pi.base_dist),
                                   pi.reinterpreted_batch_ndims)
    elif isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical`, `Independent` and '
                                  '`Normal` policies are valid policies. Got '
                                  '`{0}`.'.format(type(pi)))
    return distribution


class BaseMetaLearner(object):
    def __init__(self, policy, device='cpu'):
        self.device = torch.device(device)
        self.policy = policy
        self.policy.to(self.device)

    def step(self, train_episodes, test_episodes, *args, **kwargs):
        raise NotImplementedError()
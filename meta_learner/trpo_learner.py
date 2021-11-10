import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from base_learner import BaseMetaLearner, detach_distribution


class TRPOMetaLearner(BaseMetaLearner):
    """
    MetaLearner use TRPO algorithm, main structure comes from  github.com/tristandeleu/pytorch-maml-rl
    """

    def __init__(self, policy, fast_lr=0.5, first_order=False, device="cpu"):
        super(TRPOMetaLearner, self).__init__(policy, device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    def adapt(self, train_episodes, first_order=None):
        """
            update train episodes's params, return updated params
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

    def surrogate_loss(self, train_episodes, test_episode, old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        params = self.adapt(train_episodes, first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            pi = self.policy(test_episode.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)
            log_ratio = (pi.log_prob(test_episode.actions) - old_pi.log_prob(test_episode.actions))
            ratio = torch.exp(log_ratio)
            losses = - ratio * test_episode.advantages

            kls = kl_divergence(pi, old_pi)
        return losses.mean(), kls.mean(), old_pi

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def step(self, train_episodes, test_episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2, ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_episodes)
        surrogate_losses = [self.surrogate_loss(train_episode, test_episode)
                            for (train_episode, test_episode) in zip(train_episodes, test_episodes)]
        old_losses = [_[0] for _ in surrogate_losses]
        old_kls = [_[1] for _ in surrogate_losses]
        old_pis = [_[2] for _ in surrogate_losses]

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        # extract parameters into vectors
        grads = parameters_to_vector(grads)

        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl, cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search step size, until we reduce the loss
        step_size = 1.0
        for _ in range(ls_max_steps):
            # save the vectors into parameters in the model
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            # calculate the new loss after use updated meta params(update the log_probs here)
            surrogate_losses = [self.surrogate_loss(train_episode_pair, test_episode, old_pi)
                                for (train_episode_pair, test_episode, old_pi) in
                                zip(train_episodes, test_episodes, old_pis)]

            losses = [_[0] for _ in surrogate_losses]
            kls = [_[1] for _ in surrogate_losses]

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break

            # change step size
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()

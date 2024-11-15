from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import dlroms
from dlroms_bayesian.distributions import *


class VariationalInference(object):
    """
    Abstract class for variational inference methods.
    """
    def __init__(self, bayes: Bayesian):
        if not isinstance(bayes, Bayesian):
            raise TypeError(f"Model {torch.typename(bayes)} must be a Bayesian model.")
        self.bayes = bayes

    def update_bayes(self) -> None:
        raise NotImplementedError("The 'update_bayes' method must be implemented in a derived class.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The 'forward' method must be implemented in a derived class.")

    def train(self, mu, u, ntrain, epochs, optim, lr, lr_noise, loss=None, error=None, nvalid=0, batchsize=None):
        raise NotImplementedError("The 'train' method must be implemented in a derived class.")

    @torch.no_grad()
    def sample(self, input: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The 'sample' method must be implemented in a derived class.")


def gaussian_log_likelihood(target: torch.Tensor, output: torch.Tensor, log_beta: torch.Tensor, ntrain: int) -> torch.Tensor:
    """
    Gaussian log-likelihood (un-normalized).
    Input:
        target (torch.Tensor): target values
        output (torch.Tensor): predicted values
        beta (torch.Tensor): precision
    """
    return ntrain / output.shape[0] * (0.5 * torch.numel(target) * log_beta - 0.5 * torch.exp(log_beta) * torch.sum((target - output) ** 2))


def laplace_log_likelihood(target: torch.Tensor, output: torch.Tensor, log_beta: torch.Tensor, ntrain: int) -> torch.Tensor:
    """
    Laplace log-likelihood (un-normalized).
    """
    return ntrain / output.shape[0] * (torch.numel(target) * log_beta - torch.exp(log_beta) * torch.sum(torch.abs((target - output))))


class Bayesian(nn.Module):
    """
    Base class for Bayesian neural networks.
    Model: y = f(x, w) + n
    Noise: additive, homoscedastic (independent of input), either Gaussian or Laplace.
    """
    def __init__(self, model: dlroms.roms.ROM, noise: str = 'gaussian'):
        super(Bayesian, self).__init__()
        
        # NOTE: the ROM parent class Compound does not have a 'forward' method, so it cannot be used here
        if not isinstance(model, dlroms.roms.ROM):
            raise TypeError(f"Model {torch.typename(model)} must be a ROM model or one of its subclasses.")
        self.model = model

        # Variational inference algorithm for training
        self.trainer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prior for log precision of weights
        self.alpha_a = 1. # prior shape
        self.alpha_b = 0.05 # prior rate

        # Additive noise model
        self.beta_a = 2. # noise precision shape
        self.beta_b = 1e-6 # noise precision rate
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))
        self.log_beta.data = self.log_beta.to(self.device)

        if noise == 'gaussian':
            self.log_likelihood = gaussian_log_likelihood
        elif noise == 'laplace':
            self.log_likelihood = laplace_log_likelihood
        else:
            raise ValueError(f"Noise type {noise} is not supported.")

    def set_trainer(self, trainer: VariationalInference) -> None:
        if not isinstance(trainer, VariationalInference):
            raise TypeError(f"Trainer {torch.typename(trainer)} must be a VariationalInference instance.")
        self.trainer = trainer

    def He(self, linear=False, a=0.1, seed=None):
        """He initialization.
        """
        self.model.He(linear=linear, a=a, seed=seed) # calls the ROM 'He' method
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))

    def Hybrid(self, x1, x2):
        """Hybrid initialization.
        """
        self.model.Hybrid(x1, x2)
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))

    def cuda(self):
        """Move model to GPU.
        """
        self.device = torch.device('cuda')
        self.model.cuda()
        self.log_beta.data = self.log_beta.to(self.device)

    def cpu(self):
        """Move model to CPU.
        """
        self.device = torch.device('cpu')
        self.model.cpu()
        self.log_beta.data = self.log_beta.to(self.device)

    def count_parameters(self):
        """Count the number of parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters()) + self.log_beta.numel()

    def _log_joint(self, target: torch.Tensor, output: torch.Tensor, ntrain: int) -> torch.Tensor:
        """Compute the log joint.
        """
        log_likelihood = self.log_likelihood(target, output, self.log_beta, ntrain)
        # log Gamma(beta| a, b)
        log_prior_log_beta = (self.beta_a - 1.) * self.log_beta - torch.exp(self.log_beta) * self.beta_b
        return torch.sum(log_likelihood + log_prior_log_beta) # return a scalar

    def _log_prior(self) -> torch.Tensor:
        """Compute the log prior on the parameters (weights).
        """
        # log StudentT(w | mu, lambda = a / b, nu = 2 * a)
        log_prior_w = torch.tensor(0.0).to(self.device)
        for param in self.model.parameters():
            log_prior_w += torch.sum(torch.log1p(0.5 / self.alpha_b * param ** 2))
            log_prior_w *= - (self.alpha_a + 0.5)
        return log_prior_w

    def _log_posterior(self, target: torch.Tensor, output: torch.Tensor, ntrain: int) -> torch.Tensor:
        """Compute the un-normalized log posterior."""
        log_joint = self._log_joint(target, output, ntrain)
        log_prior = self._log_prior()
        return log_joint + log_prior

    def forward(self, *args, **kwargs):
        """Forward pass through the model (each pass gives a different stochastic output).
        """
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'forward' method.")
        return self.trainer.forward(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'train' method.")
        return self.trainer.train(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'sample' method.")
        return self.trainer.sample(*args, **kwargs)

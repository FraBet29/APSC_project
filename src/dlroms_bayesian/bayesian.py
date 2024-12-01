from __future__ import annotations
from typing import *

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import dlroms
from dlroms_bayesian.utils import *
from dlroms_bayesian.expansions import *
from dlroms_bayesian.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VariationalInference(ABC):
    """
    Abstract class for variational inference methods.
    """
    def __init__(self, bayes: Bayesian):
        """
        Initialize the variational inference algorithm.
        Args:
            bayes (Bayesian): Bayesian model
        """
        if not isinstance(bayes, Bayesian):
            raise TypeError(f"Model {torch.typename(bayes)} must be a Bayesian model.")
        self.bayes = bayes

    @abstractmethod
    def update(self) -> None:
        """
        Update the Bayesian model.
        """
        pass

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the Bayesian model.
        Args:
            input (torch.Tensor): input data
        Returns:
            torch.Tensor: output data
        """
        pass

    @abstractmethod
    def train(self, mu: torch.Tensor, u: torch.Tensor, ntrain: int, epochs: int, optim: torch.optim.Optimizer, 
              lr: float, lr_noise: float, loss: Callable, error: Optional[Callable] = None, nvalid: int = 0) -> None:
        """
        Train the Bayesian model.
        Args:
            mu (torch.Tensor): inputs
            u (torch.Tensor): targets
            ntrain (int): number of training samples
            epochs (int): number of training epochs
            optim (torch.optim.Optimizer): optimizer
            lr (float): learning rate
            lr_noise (float): learning rate for the noise
            loss (Callable): loss function
            error (Optional[Callable]): error function (default: None)
            nvalid (int): number of validation samples (default: 0)
        Returns:
            None
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def sample(self, input: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the Bayesian model.
        Args:
            input (torch.Tensor): input data
            n_samples (int): number of samples
        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and variance of the samples
        """
        pass


def gaussian_log_likelihood(target: torch.Tensor, output: torch.Tensor, log_beta: torch.Tensor, ntrain: int) -> torch.Tensor:
    """
    Gaussian log-likelihood (un-normalized).
    Args:
        target (torch.Tensor): target values
        output (torch.Tensor): predicted values
        log_beta (torch.Tensor): log precision
        ntrain (int): number of training samples
    Returns:
        torch.Tensor: un-normalized log-likelihood
    """
    return ntrain / output.shape[0] * (0.5 * torch.numel(target) * log_beta - 0.5 * torch.exp(log_beta) * torch.sum((target - output) ** 2))


def laplace_log_likelihood(target: torch.Tensor, output: torch.Tensor, log_beta: torch.Tensor, ntrain: int) -> torch.Tensor:
    """
    Laplace log-likelihood (un-normalized).
    Args:
        target (torch.Tensor): target values
        output (torch.Tensor): predicted values
        log_beta (torch.Tensor): log precision
        ntrain (int): number of training samples
    Returns:
        torch.Tensor: un-normalized log-likelihood
    """
    return ntrain / output.shape[0] * (torch.numel(target) * log_beta - torch.exp(log_beta) * torch.sum(torch.abs((target - output))))


class Bayesian(nn.Module):
    """
    Base class for Bayesian neural networks.
    Model:
        y = f(x, w) + n,
    where f is the neural network, x is the input, w are the weights, and n is the noise (additive and homoscedastic).
    """
    def __init__(self, model: dlroms.roms.ROM, noise: str = 'gaussian'):
        """
        Initialize the Bayesian model.
        Args:
            model (dlroms.roms.ROM): reduced-order model
            noise (str): type of noise (default: 'gaussian')
        """
        super(Bayesian, self).__init__()
        
        # NOTE: the ROM parent class Compound does not have a 'forward' method, so it cannot be used here
        if not isinstance(model, dlroms.roms.ROM):
            raise TypeError(f"Model {torch.typename(model)} must be a ROM model or one of its subclasses.")
        self.model = model

        # Variational inference algorithm for training
        self.trainer = None

        # Prior for log precision of weights
        self._alpha_a = 1. # prior shape
        self._alpha_b = 0.05 # prior rate

        # Additive noise model
        self._beta_a = 2. # noise precision shape
        self._beta_b = 1e-6 # noise precision rate
        self.log_beta = Parameter(torch.log(Gamma(self._beta_a, self._beta_b).sample((1,))))

        if noise == 'gaussian':
            self.log_likelihood = gaussian_log_likelihood
        elif noise == 'laplace':
            self.log_likelihood = laplace_log_likelihood
        else:
            raise ValueError(f"Noise type {noise} is not supported.")

    def set_trainer(self, trainer: VariationalInference) -> None:
        """
        Set the variational inference algorithm for training.
        Args:
            trainer (VariationalInference): variational inference algorithm
        Returns:
            None
        """
        if not isinstance(trainer, VariationalInference):
            raise TypeError(f"Trainer {torch.typename(trainer)} must be a VariationalInference instance.")
        self.trainer = trainer

    @torch.no_grad
    def _reset_log_beta(self):
        """
        Sample a new value for the log precision of the noise.
        """
        self.log_beta.copy_(torch.log(Gamma(self._beta_a, self._beta_b).sample((1,))))

    def He(self, linear=False, a=0.1, seed=None):
        """
        He initialization (see DLROMs Layer class).
        """
        self.model.He(linear=linear, a=a, seed=seed) # calls the ROM 'He' method
        self._reset_log_beta()

    def deterministic(self, x1, x2):
        """
        Deterministic initialization.
        """
        self.model.deterministic(x1, x2)
        self._reset_log_beta()

    def hybrid(self, x1, x2):
        """
        Hybrid initialization.
        """
        self.model.hybrid(x1, x2)
        self._reset_log_beta()

    def cuda(self):
        """
        Move model to GPU.
        """
        self.model.cuda()
        self.log_beta = self.log_beta.to(torch.device('cuda'))

    def cpu(self):
        """
        Move model to CPU.
        """
        self.model.cpu()
        self.log_beta = self.log_beta.to(torch.device('cpu'))

    def count_parameters(self):
        """
        Count the number of parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters()) + self.log_beta.numel()

    def _log_joint(self, target: torch.Tensor, output: torch.Tensor, ntrain: int) -> torch.Tensor:
        """
        Compute the log joint.
        Args:
            target (torch.Tensor): target values
            output (torch.Tensor): predicted values
            ntrain (int): number of training samples
        Returns:
            torch.Tensor: un-normalized log joint
        """
        log_likelihood = self.log_likelihood(target, output, self.log_beta, ntrain)
        log_prior_log_beta = (self._beta_a - 1.) * self.log_beta - torch.exp(self.log_beta) * self._beta_b # log Gamma(beta|a, b)
        return torch.sum(log_likelihood + log_prior_log_beta) # return a scalar

    def _log_prior(self) -> torch.Tensor:
        """
        Compute the log prior on the parameters.
        Returns:
            torch.Tensor: un-normalized log prior
        """
        log_prior_w = torch.tensor(0.0).to(device)
        for param in self.model.parameters():
            log_prior_w += torch.sum(torch.log1p(0.5 / self._alpha_b * param ** 2))
        log_prior_w *= - (self._alpha_a + 0.5) # log Student-t
        return log_prior_w

    def _log_posterior(self, target: torch.Tensor, output: torch.Tensor, ntrain: int) -> torch.Tensor:
        """
        Compute the un-normalized log posterior.
        Args:
            target (torch.Tensor): target values
            output (torch.Tensor): predicted values
            ntrain (int): number of training samples
        Returns:
            torch.Tensor: un-normalized log posterior
        """
        log_joint = self._log_joint(target, output, ntrain)
        log_prior = self._log_prior()
        return log_joint + log_prior

    def forward(self, *args, **kwargs):
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

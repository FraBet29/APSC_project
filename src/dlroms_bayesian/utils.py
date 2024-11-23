from typing import Callable

import random
import numpy as np
import os
import torch
import torch.distributions

from dlroms.roms import mse


def set_seeds(seed: int, deterministic: bool = True) -> None:
	"""
	Set seeds for reproducibility.
	Args:
		seed: integer seed.
		deterministic: whether to set deterministic mode.
	Returns:
		None
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True, warn_only=True)
		os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def multi_mse(norm: Callable) -> Callable:
    """
    Compute the MSE, with optional reduction across the output channels.
    Args:
        norm: function to compute the norm of the error.
    Returns:
        mse_fn: function to compute the MSE across the output channels.
    """
    def mse_fn(utrue: torch.Tensor, upred: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        results = torch.stack([norm(utrue[:, c] - upred[:, c], squared=True) for c in range(utrue.shape[1])], dim=1)
        return results.mean() if reduce else results.mean(0)
    return mse_fn


def multi_mre(norm: Callable) -> Callable:
    """
    Compute the MRE, with optional reduction across the output channels.
    Args:
        norm: function to compute the norm of the error.
    Returns:
        mre_fn: function to compute the MRE across the output channels.
    """
    def mre_fn(utrue: torch.Tensor, upred: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        results = torch.stack([(norm(utrue[:, c] - upred[:, c]) / norm(utrue[:, c])) for c in range(utrue.shape[1])], dim=1)
        return results.mean() if reduce else results.mean(0)
    return mre_fn


def rsquared(norm: Callable) -> Callable:
    """
    Compute the R-squared coefficient of determination.
    Args:
        norm: function to compute the norm of the error.
    Returns:
        rsquared_fn: function to compute the R-squared coefficient of determination.
    """
    def rsquared_fn(utrue: torch.Tensor, upred: torch.Tensor) -> torch.Tensor:
        return 1. - mse(norm)(utrue, upred) / mse(norm)(utrue, utrue.mean(0))
    return rsquared_fn


class Gaussian(object):
    """
    Gaussian distribution. It builds upon the PyTorch Normal distribution, but with a specific log_prob method, namely:
        If 'x' is a scalar, it behaves as the PyTorch Normal distribution;
        If 'x' is a n-dimensional tensor, it computes the log probability of i.i.d. samples.
    """
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        log_prob = torch.distributions.Normal(self.mu, self.sigma).log_prob(x)
        if isinstance(x, (int, float)) or x.dim() < 1:
            return log_prob # 'x' is a scalar
        else:
            return torch.sum(log_prob) # 'x' is a n-dimensional tensor of i.i.d. samples

    def sample(self, shape):
        return torch.distributions.Normal(self.mu, self.sigma).sample(shape)


class Laplace(object):
    """
    Laplace distribution. It builds upon the PyTorch Laplace distribution, but with a specific log_prob method, namely:
        If 'x' is a scalar, it behaves as the PyTorch Laplace distribution;
        If 'x' is a n-dimensional tensor, it computes the log probability of i.i.d. samples.
    """
    def __init__(self, mu=0., b=1.):
        self.mu = mu
        self.b = b

    def log_prob(self, x):
        log_prob = torch.distributions.Laplace(self.mu, self.b).log_prob(x)
        if isinstance(x, (int, float)) or x.dim() < 1:
            return log_prob
        else:
            return torch.sum(log_prob)

    def sample(self, shape):
        return torch.distributions.Laplace(self.mu, self.b).sample(shape)


class Gamma(object):
    """
    Gamma distribution (see PyTorch Gamma distribution).
    """
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        return torch.distributions.Gamma(self.alpha, self.beta).log_prob(x)

    def sample(self, shape):
        return torch.distributions.Gamma(self.alpha, self.beta).sample(shape)


class StudentT(object):
    """
    Student's t distribution (see PyTorch Student's t distribution).
    """
    def __init__(self, mu=0., sigma=1., nu=1.):
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    def log_prob(self, x):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).log_prob(x)

    def sample(self, shape):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).sample(shape)
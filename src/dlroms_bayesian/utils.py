import torch
import torch.distributions
from dlroms.roms import mse


def multi_mse(norm):
    """
    Returns a function to compute MSE, with optional reduction across the output channels.
    """
    def mse_fn(utrue, upred, reduce=True):
        results = torch.stack([norm(utrue[:, c] - upred[:, c], squared=True) for c in range(utrue.shape[1])], dim=1)
        return results.mean() if reduce else results.mean(0)
    return mse_fn


def multi_mre(norm):
    """
    Returns a function to compute MRE, with optional reduction across the output channels.
    """
    def mre_fn(utrue, upred, reduce=True):
        results = torch.stack([(norm(utrue[:, c] - upred[:, c]) / norm(utrue[:, c])) for c in range(utrue.shape[1])], dim=1)
        return results.mean() if reduce else results.mean(0)
    return mre_fn


def rsquared(norm):
    """
    Returns a function to compute the R-squared coefficient of determination.
    """
    def rsquared_fn(utrue, upred):
        return 1. - mse(norm)(utrue, upred) / mse(norm)(utrue, utrue.mean(0))


class Gaussian(object):
    """Gaussian distribution with mean 'mu' and standard deviation 'sigma'."""
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
    """Laplace distribution with mean 'mu' and scale 'b'."""
    def __init__(self, mu=0., b=1.):
        self.mu = mu
        self.b = b

    def log_prob(self, x):
        return torch.distributions.Laplace(self.mu, self.b).log_prob(x)

    def sample(self, shape):
        return torch.distributions.Laplace(self.mu, self.b).sample(shape)


class Gamma(object):
    """Gamma distribution with shape 'alpha' and rate 'beta'."""
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        return torch.distributions.Gamma(self.alpha, self.beta).log_prob(x)

    def sample(self, shape):
        return torch.distributions.Gamma(self.alpha, self.beta).sample(shape)


class StudentT(object):
    """Student's t distribution with mean 'mu', scale 'sigma', and degrees of freedom 'nu'."""
    def __init__(self, mu=0., sigma=1., nu=1.):
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    def log_prob(self, x):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).log_prob(x)

    def sample(self, shape):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).sample(shape)
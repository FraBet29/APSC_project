import torch
import torch.nn as nn
import torch.distributions
from torch.nn.parameter import Parameter

import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm

from minns import *
from roms import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: get device from ROM?


###############
### Helpers ###
###############

def parameters_to_vector(parameters, grad=True):
    """Convert parameters (and their gradients) to a tensor."""
    if not grad:
        vec_params = []
        for param in parameters:
            vec_params.append(param.data.view(-1))
        return torch.cat(vec_params)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)


def vector_to_parameters(vec, parameters, grad=True):
    """Convert one vector to the parameters or gradients of the parameters."""
    idx = 0
    if grad:
        for param in parameters:
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[idx:idx+num_param].view(param.size())
            idx += num_param
    else:
        for param in parameters:
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[idx:idx+num_param].view(param.size())
            idx += num_param


def gaussian_log_likelihood(target, output, beta):
    """
    Gaussian log-likelihood (un-normalized).
    Input:
        target (torch.Tensor): target values
        output (torch.Tensor): predicted values
        beta (torch.Tensor): precision
    """
    return torch.sum(- 0.5 * torch.numel(target) * torch.log(beta) - 0.5 * beta * (target - output) ** 2)


def laplace_log_likelihood(target, output, beta):
    """
    Laplace log-likelihood (un-normalized).
    """
    return torch.sum(- torch.numel(target) * torch.log(beta) - beta * torch.abs((target - output)))


##########################################
### Wrappers for PyTorch distributions ###
##########################################

class Gaussian(object):
    """Gaussian distribution with mean 'mu' and standard deviation 'sigma'."""
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        return torch.distributions.MultivariateNormal(self.mu, self.sigma).log_prob(x)

    def sample(self, shape):
        return torch.distributions.MultivariateNormal(self.mu, self.sigma).sample(shape)


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


#########################################
### Classes for variational inference ###
#########################################

class VariationalInference(object):
    """
    Abstract class for variational inference methods.
    """
    def __init__(self, bayes):
        if not isinstance(bayes, Bayesian):
            raise TypeError(f"Model {torch.typename(bayes)} must be a Bayesian model.")
        self.bayes = bayes

    def forward(self, input):
        pass

    def train(self):
        pass

    def sample(self, n_samples):
        pass


class SVI(VariationalInference):
    """
    Stochastic variatiational inference based on the maximization of the ELBO.
    """
    def __init__(self):
        super(SVI, self).__init__()

    def train(self):
        pass

    def sample(self):
        pass


def RBF_kernel(X, h=-1):

    X_norm_squared = torch.sum(X ** 2, dim=1, keepdim=True)
    pairwise_dists_squared = X_norm_squared + X_norm_squared.T - 2 * torch.mm(X, X.T)

    if h < 0: # if h < 0, use median trick
        h = torch.median(pairwise_dists_squared)
        h = math.sqrt(0.5 * h / math.log(X.shape[0]))

    Kxx = torch.exp(-0.5 * pairwise_dists_squared / h ** 2)
    dxKxx = (torch.diag(torch.sum(Kxx, 1)) - Kxx) @ X / (h ** 2) # TODO: check if this is correct

    return Kxx, dxKxx


class SVGD(VariationalInference):
    """
    Stein variational gradient descent.
    """
    def __init__(self, bayes, n_samples=20, kernel='rbf', lr=0.01, lr_noise=0.01):
        super(SVGD, self).__init__(bayes)

        self.n_samples = n_samples
        
        if kernel == 'rbf':
            self.kernel = RBF_kernel
        else:
            raise ValueError(f"Kernel type {kernel} is not supported.")

        # Store n_samples instances of the model
        models = []
        for _ in range(n_samples):
            model = deepcopy(bayes)
            model.He() # He initialization # TODO: add option to choose initialization
            models.append(model)
        self.models = models # list of Bayesian models
        del models

        # Store n_samples instances of the optimizer
        # NOTE: LBFGS does not support per-parameter options and parameter groups
        optimizers = []
        for idx in range(n_samples):
            parameters = [{'params': self.models[idx].model.parameters()}, # parameters of the ROM model
                          {'params': [self.models[idx].log_beta], 'lr': lr_noise}]
            optimizer = torch.optim.Adam(parameters, lr=lr)
            optimizers.append(optimizer)
        self.optimizers = optimizers
        del optimizers

    def __getitem__(self, idx):
        return self.models[idx]

    def forward(self, input):
        """
        Returns the output of the ensemble of models.
        """
        outputs = []
        for idx in range(self.n_samples):
            outputs.append(self.models[idx].model.forward(input)) # calls the ROM 'forward()' method
        outputs = torch.stack(outputs)
        return outputs
    
    def __call__(self, input):
        """
        Returns the average output of the ensemble of models.
        """
        output = torch.zeros_like(input)
        for idx in range(self.n_samples):
            output += self.models[idx].model.forward(input) # calls the ROM 'forward()' method
        return output / self.n_samples

    # TODO: replace for loop with torch.stack?
    def train(self, mu, u, ntrain, epochs, loss=None, error=None, nvalid=0, batchsize=None):

        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise ValueError("The trainer must be set in the Bayesian model before training.")

        # TODO: pass optimizer here (as done in ROM)?
        # TODO: support batch computation
        if batchsize is not None:
            raise NotImplementedError("Batch computation is not supported.")

        M = (mu,) if(isinstance(mu, torch.Tensor)) else (mu if (isinstance(mu, tuple)) else None) # NOTE: same as ROM
        U = (u,) if(isinstance(u, torch.Tensor)) else (u if (isinstance(u, tuple)) else None)

        if(M == None):
            raise RuntimeError("Input data should be either a torch.Tensor or a tuple of torch.Tensors.")
        if(U == None):
            raise RuntimeError("Output data should be either a torch.Tensor or a tuple of torch.Tensors.")

        ntest = len(U[0]) - ntrain
        Mtrain, Utrain = tuple([m[:(ntrain-nvalid)] for m in M]), tuple([um[:(ntrain-nvalid)] for um in U])
        Mvalid, Uvalid = tuple([m[(ntrain-nvalid):ntrain] for m in M]), tuple([um[(ntrain-nvalid):ntrain] for um in U])
        Mtest, Utest = tuple([m[-ntest:] for m in M]), tuple([um[-ntest:]for um in U])

        getout = (lambda y: y[0]) if len(U) == 1 else (lambda y: y)
        errorf = (lambda a, b: error(a, b)) if error != None else (lambda a, b: loss(a, b))
        validerr = (lambda : np.nan) if nvalid == 0 else (lambda : errorf(getout(Uvalid), self(*Mvalid)).item())
        testerr = (lambda : np.nan) if ntest == 0 else (lambda : errorf(getout(Utest), self(*Mtest)).item())

        err = []

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):

                for model in self.models:
                    model.zero_grad() # TODO: check if all gradients are zeroed

                grad_log_posterior = [] # all gradients of log posterior probability: (S, P)
                theta = [] # all model parameters (particles): (S, P)

                Upred = torch.zeros_like(getout(Utrain))

                for i in range(self.n_samples):

                    Upred_i = self.models[i].model.forward(*(Mtrain)) # call the ROM 'forward()' method with multiple inputs
                    Upred += Upred_i.detach()

                    log_posterior_i = self.models[i].log_posterior(Upred_i, getout(Utrain), ntrain)
                    log_posterior_i.backward() # compute gradients of log posterior

                    vec_param, vec_grad_log_posterior = parameters_to_vector(self.models[i].parameters(), grad=True)
                    grad_log_posterior.append(torch.unsqueeze(vec_grad_log_posterior, 0)) # concatenate log joint gradients
                    theta.append(torch.unsqueeze(vec_param, 0)) # concatenate parameters

                theta = torch.cat(theta)
                grad_log_posterior = torch.cat(grad_log_posterior)

                ### SVGD update ###
                Kxx, dxKxx = self.kernel(theta)
                grad_theta = - (torch.mm(Kxx, grad_log_posterior) + dxKxx) / self.n_samples
            
                for i in range(self.n_samples):
                    vector_to_parameters(grad_theta[i], self.models[i].parameters(), grad=True)
                    self.optimizers[i].step()
                
                err.append([errorf(getout(Utrain), self(*Mtrain)).item(), validerr(), testerr()])

                # Early stopping
                if(nvalid > 0 and len(err) > 3): # check if validation error is increasing
                    if((err[-1][1] > err[-2][1]) and (err[-1][0] < err[-2][0])):
                        if((err[-2][1] > err[-3][1]) and (err[-2][0] < err[-3][0])):
                            print("Early stopping.")
                            break

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {err[-1][0]:.6f}, valid: {err[-1][1]} test: {err[-1][2]:.6f}")
                pbar.update()

    @torch.no_grad()
    def sample(self, input, n_samples):
        if n_samples > self.n_samples:
            raise ValueError(f"The number of samples ({n_samples}) exceeds the number of instances ({self.n_samples}).")

        outputs = self.forward(input)
        outputs = outputs[:n_samples]

        output_mean = torch.mean(outputs, dim=0) # E[y + n] = E[y] since the additive noise has zero mean
        output_var = torch.var(outputs, dim=0)
        beta = torch.exp(self.bayes.log_beta)
        noise_var = beta.data.item() # TODO: check if beta has been updated

        return output_mean, output_var + noise_var


###############################
### Bayesian neural network ###
###############################

# TODO: add device selection
# TODO: add freeze method (requires_grad=False)

class Bayesian(nn.Module):
    """
    Base class for Bayesian neural networks.

    Model: y = f(x, w) + n

    Noise: additive, homoscedastic (independent of input), either Gaussian or Laplace.
        n ~ Gaussian(0, 1 / beta) or n ~ Laplace(0, 1 / beta)
        beta ~ Gamma(beta | a, b)
    """
    def __init__(self, model, noise='gaussian'):
        super(Bayesian, self).__init__()
        
        # NOTE: the ROM parent class Compound does not have a 'forward' method, so it cannot be used here
        if not isinstance(model, ROM):
            raise TypeError(f"Model {torch.typename(model)} must be a ROM model or one of its subclasses.")
        self.model = model

        # Prior for log precision of weights
        self.alpha_a = 1. # prior shape
        self.alpha_b = 0.05 # prior rate

        # Additive noise model
        self.beta_a = 2. # noise precision shape
        self.beta_b = 1e-6 # noise precision rate
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,)))) # TODO: set device

        if noise == 'gaussian':
            self.log_likelihood = gaussian_log_likelihood
        elif noise == 'laplace':
            self.log_likelihood = laplace_log_likelihood
        else:
            raise ValueError(f"Noise type {noise} is not supported.")

        self.trainer = None

    def set_trainer(self, trainer):
        if not isinstance(trainer, VariationalInference):
            raise TypeError(f"Trainer {torch.typename(trainer)} must be a VariationalInference instance.")
        self.trainer = trainer

    @torch.no_grad()
    def He(self, linear=False, a=0.1, seed=None):
        """He initialization.
        """
        self.model.He(linear=linear, a=a, seed=seed) # calls the ROM 'He' method
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,)))) # TODO: set device

    @torch.no_grad()
    def Xavier(self):
        """Xavier initialization.
        """
        self.model.Xavier() # calls the ROM 'Xavier' method
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))

    def forward(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'forward' method.")
        return self.trainer.forward(*args, **kwargs)

    # TODO: replace analytical log posterior with numerical approximation?
    def log_posterior(self, target, output, ntrain):

        log_likelihood = self.log_likelihood(target, output, torch.exp(self.log_beta)) # TODO: normalization constant ntrain / target.size[0]?

        # log StudentT(w | mu, lambda = a / b, nu = 2 * a)
        log_prior_w = torch.tensor(0.0).to(device) # TODO: where is device defined?
        for param in self.model.parameters():
            log_prior_w += torch.sum(torch.log1p(0.5 / self.alpha_b * param ** 2))
            log_prior_w *= - (self.alpha_a + 0.5)

        # log Gamma(beta| a, b)
        log_prior_log_beta = (self.beta_a - 1.) * self.log_beta - torch.exp(self.log_beta) * self.beta_b
        log_prior_log_beta = torch.sum(log_prior_log_beta) # return a scalar

        return log_likelihood + log_prior_w + log_prior_log_beta

    def train(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'train' method.")
        return self.trainer.train(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'sample' method.")
        return self.trainer.sample(*args, **kwargs)

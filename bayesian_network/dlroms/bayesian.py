import torch
import torch.distributions
from torch.nn.parameter import Parameter

import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm

from minns import *
from roms import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: get device from ROM?


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
    """Convert one vector to the parameters or gradients of the parameters
    Arguments:
        vec (torch.Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): True for assigning de-vectorized `vec` to gradients
    """
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


# Wrappers for PyTorch distributions

class Gaussian(object):
    """Gaussian distribution with mean `mu` and standard deviation `sigma`."""
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        return torch.distributions.MultivariateNormal(self.mu, self.sigma).log_prob(x)

    def sample(self, shape):
        return torch.distributions.MultivariateNormal(self.mu, self.sigma).sample(shape)


class Laplace(object):
    """Laplace distribution with mean `mu` and scale `b`."""
    def __init__(self, mu=0., b=1.):
        self.mu = mu
        self.b = b

    def log_prob(self, x):
        return torch.distributions.Laplace(self.mu, self.b).log_prob(x)

    def sample(self, shape):
        return torch.distributions.Laplace(self.mu, self.b).sample(shape)


class Gamma(object):
    """Gamma distribution with shape `alpha` and rate `beta`."""
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        return torch.distributions.Gamma(self.alpha, self.beta).log_prob(x)

    def sample(self, shape):
        return torch.distributions.Gamma(self.alpha, self.beta).sample(shape)


class StudentT(object):
    """Student's t distribution with mean `mu`, scale `sigma`, and degrees of freedom `nu`."""
    def __init__(self, mu=0., sigma=1., nu=1.):
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    def log_prob(self, x):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).log_prob(x)

    def sample(self, shape):
        return torch.distributions.StudentT(self.nu, self.mu, self.sigma).sample(shape)


# Main class for Bayesian Neural Networks

# TODO: add device selection
# TODO: add freeze method (requires_grad=False)
class Bayesian(object): # TODO: base class?
    """Class for Bayesian NNs with Stein Variational Gradient Descent.

    Bayesian NNs: y = f(x, w) + n

    ### TODO: change this using the prior from the mesh-informed layer ###
    Weights:
        w_i ~ Normal(w_i | mu, 1 / alpha)
        alpha ~ Gamma(alpha | a, b) (shared)
        by marginalizing out alpha: w_i ~ StudentT(w_i | mu, lambda = a / b, nu = 2 * a)

    Noise: additive, homoscedastic (independent of input), output-wise, either Gaussian or Laplace.
        n ~ Normal(0, 1 / beta) or n ~ Laplace(0, 1 / beta)
        beta ~ Gamma(beta | a, b)
    """
    def __init__(self, model, n_samples=20):

        super(Bayesian, self).__init__()
        if not isinstance(model, ROM):
            raise TypeError(f"model {torch.typename(model)} is not a ROM subclass.")
        # NOTE: the ROM parent class Compound does not have a 'forward' method

        self.n_samples = n_samples

        # store n_samples instances of the model
        instances = []
        for i in range(n_samples):
            instance = deepcopy(model)
            instances.append(instance)
        self.nets = instances # list of ROMs
        del instances

        # prior for log precision of weights
        self.alpha_a = 1. # prior shape
        self.alpha_b = 0.05 # prior rate

        # prior for log precision of Gaussian or Laplace noise
        self.beta_a = 2. # noise precision shape
        self.beta_b = 1e-6 # noise precision rate
        log_beta = Gamma(self.beta_a, self.beta_b).sample((self.n_samples,)).log()
        for i in range(n_samples):
            self.nets[i].log_beta = Parameter(log_beta[i]) # added automatically to the list of parameters

    @torch.no_grad()
    def He(self, linear=False, a=0.1, seed=None):
        """He initialization of the weights for each model instance.
        """
        for model in self.nets:
            model.He(linear=linear, a=a, seed=seed) # calls the ROM 'He' method

    @torch.no_grad()
    def Xavier(self):
        """Xavier initialization of the weights for each model instance.
        """
        for model in self.nets:
            model.Xavier() # calls the ROM 'Xavier' method

    @property
    def parameters(self): # TODO: should return an iterator?
        """Returns the parameters of the Bayesian NNs."""
        params = []
        for model in self.nets:
            params += list(model.parameters()) + [model.log_beta]
        return params

    @property
    def log_beta(self):
        """Returns the log precision of the Gaussian or Laplace noise."""
        return torch.tensor([net.log_beta.item() for net in self.nets], device=device)

    def __getitem__(self, idx):
        return self.nets[idx]

    def forward(self, input):
        output = []
        for i in range(self.n_samples):
            output.append(self.nets[i].forward(input)) # calls the ROM 'forward()' method
        output = torch.stack(output)
        return output

    # TODO: replace analytical log joint with numerical approximation?
    def log_joint(self, index, output, target, ntrain):
        """Computes the log joint probability for each model instance in the Bayesian NNs.

        Returns:
            Log joint probability (zero-dim tensor)
        """
        # log Normal(target | output, 1 / beta * I)
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nets[index].log_beta.exp()
                            * (target - output).pow(2).sum()
                            + 0.5 * target.numel() * self.nets[index].log_beta)

        # log Laplace(target | output, 1 / beta * I)
        # log_likelihood = ntrain / output.size(0) * (
        #                    - self.nets[index].log_beta.exp()
        #                    * (target - output).abs().sum()
        #                    + target.numel() * self.nets[index].log_beta)

        # log StudentT(w | mu, lambda, nu)
        log_prob_prior_w = torch.tensor(0.0).to(device)
        for param in self.nets[index].parameters():
            log_prob_prior_w += torch.log1p(0.5 / self.alpha_b * param.pow(2)).sum()
            log_prob_prior_w *= - (self.alpha_a + 0.5)

        # log Gamma(beta | a, b)
        log_prior_log_beta = ((self.beta_a - 1.) * self.nets[index].log_beta
                              - self.nets[index].log_beta.exp() * self.beta_b)

        return log_likelihood + log_prob_prior_w + log_prior_log_beta

    @torch.no_grad()
    def predict(self, x):
        """Predictive mean and variance of the Bayesian NNs."""
        y = self.forward(x) # the 'forward' method stacks the outputs from the different model instances

        y_pred_mean = y.mean(0) # mean over the different instances

        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (- self.log_beta).exp()
        y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y_pred_mean, y_pred_var


# TODO: implement with batch computation
class SVGD(object):
    """Base class for Stein Variational Gradient Descent.
    """
    def __init__(self, bayes_nn, n_samples=20, optim=torch.optim.Adam, lr=0.01, lr_noise=0.01):
        
        self.bayes_nn = bayes_nn
        self.n_samples = n_samples

        optimizers = []

        for i in range(self.n_samples):
            parameters = [{'params': self.bayes_nn[i].parameters()},
                          {'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise}]
            optimizer = optim(parameters, lr=lr) # NOTE: LBFGS does not support per-parameter options and parameter groups
            optimizers.append(optimizer)
        
        self.optimizers = optimizers
        del optimizers

    # TODO: support other kernels?
    def Kxx_dxKxx(self, X, h=-1):
        """
        Computes covariance matrix K(X, X) and its gradient w.r.t. X for RBF kernel.

        Args:
            X (Tensor): (S, P), where S is num of samples and P is the dim of one sample.
        """
        XXT = torch.mm(X, X.T)
        XTX = XXT.diag()
        squared_dist = -2.0 * XXT + XTX + XTX.unsqueeze(1)

        if h < 0: # if h < 0, use median trick
            h = squared_dist.median()
            h = torch.sqrt(0.5 * h / math.log(self.n_samples))

        Kxx = torch.exp(-0.5 * squared_dist / h ** 2) # RBF kernel
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / h ** 2

        return Kxx, dxKxx

    def train(self, mu, u, ntrain, epochs, loss=None, error=None, nvalid=0, batchsize=None):

        M = (mu,) if(isinstance(mu, torch.Tensor)) else (mu if (isinstance(mu, tuple)) else None) # NOTE: same as ROM
        U = (u,) if(isinstance(u, torch.Tensor)) else (u if (isinstance(u, tuple)) else None)
        # M = mu if(isinstance(mu, torch.Tensor)) else None
        # U = u if(isinstance(u, torch.Tensor)) else None

        if(M == None):
            raise RuntimeError("Input data should be either a torch.Tensor or a tuple of torch.Tensors.")
        if(U == None):
            raise RuntimeError("Output data should be either a torch.Tensor or a tuple of torch.Tensors.")

        ntest = len(U[0]) - ntrain
        Mtrain, Utrain = tuple([m[:(ntrain-nvalid)] for m in M]), tuple([um[:(ntrain-nvalid)] for um in U])
        Mvalid, Uvalid = tuple([m[(ntrain-nvalid):ntrain] for m in M]), tuple([um[(ntrain-nvalid):ntrain] for um in U])
        Mtest, Utest = tuple([m[-ntest:] for m in M]), tuple([um[-ntest:]for um in U])
        # Mtrain, Utrain = M[:(ntrain-nvalid)], U[:(ntrain-nvalid)]
        # Mvalid, Uvalid = M[(ntrain-nvalid):ntrain], U[(ntrain-nvalid):ntrain]
        # Mtest, Utest = M[-ntest:], U[-ntest:]

        getout = (lambda y: y[0]) if len(U) == 1 else (lambda y: y)
        errorf = (lambda a, b: error(a, b)) if error != None else (lambda a, b: loss(a, b))

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):

                for model in self.bayes_nn.nets:
                    model.zero_grad() # TODO: check if all gradients are zeroed (especially for log_beta)

                grad_log_joint = [] # all gradients of log joint probability: (S, P)
                theta = [] # all model parameters (particles): (S, P)

                Upred = torch.zeros_like(getout(Utrain))

                for i in range(self.n_samples):

                    Upred_i = self.bayes_nn[i].forward(*(Mtrain)) # call the ROM 'forward()' method with multiple inputs
                    Upred += Upred_i.detach()

                    log_joint_i = self.bayes_nn.log_joint(i, Upred_i, getout(Utrain), ntrain)
                    log_joint_i.backward() # compute gradients of log joint probabilities

                    vec_param, vec_grad_log_joint = parameters_to_vector(self.bayes_nn[i].parameters(), grad=True)
                    grad_log_joint.append(vec_grad_log_joint.unsqueeze(0)) # concatenate log joint gradients
                    theta.append(vec_param.unsqueeze(0)) # concatenate parameters

                theta = torch.cat(theta)
                grad_log_joint = torch.cat(grad_log_joint)

                ### SVGD update ###
                Kxx, dxKxx = self.Kxx_dxKxx(theta)
                grad_theta = - (torch.mm(Kxx, grad_log_joint) + dxKxx) / self.n_samples
            
                for i in range(self.n_samples):
                    vector_to_parameters(grad_theta[i], self.bayes_nn[i].parameters(), grad=True)
                    self.optimizers[i].step()
                
                error_train = errorf(Upred / self.n_samples, getout(Utrain)).item() # TODO: early stopping?

                # compute test (validation?) error
                if ntest > 0:
                    Ucheck = torch.zeros_like(getout(Utest))

                    for i in range(self.n_samples):
                        Ucheck_i = self.bayes_nn[i].forward(*(Mtest))
                        Ucheck += Ucheck_i.detach()
                    
                    error_test = errorf(Ucheck / self.n_samples, getout(Utest)).item()

                if ntest > 0:
                    pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {error_train:.6f}, test: {error_test:.6f}")
                else:
                    pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {error_train:.6f}, test: nan")
                pbar.update()

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
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[idx:idx+num_param].view(param.size())
            idx += num_param
    else:
        for param in parameters:
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[idx:idx+num_param].view(param.size())
            idx += num_param


def gaussian_log_likelihood(target, output, log_beta, ntrain):
    """
    Gaussian log-likelihood (un-normalized).
    Input:
        target (torch.Tensor): target values
        output (torch.Tensor): predicted values
        beta (torch.Tensor): precision
    """
    return ntrain / output.shape[0] * (0.5 * torch.numel(target) * log_beta - 0.5 * torch.exp(log_beta) * torch.sum((target - output) ** 2))


def laplace_log_likelihood(target, output, log_beta, ntrain):
    """
    Laplace log-likelihood (un-normalized).
    """
    return ntrain / output.shape[0] * (torch.numel(target) * log_beta - torch.exp(log_beta) * torch.sum(torch.abs((target - output))))


##########################################
### Wrappers for PyTorch distributions ###
##########################################

class Gaussian(object):
    """Gaussian distribution with mean 'mu' and standard deviation 'sigma'."""
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        log_prob = torch.distributions.Normal(self.mu, self.sigma).log_prob(x)
        if isinstance(x, (int, float)) or x.dim() < 1:
            return log_prob
        else:
            return torch.sum(log_prob) # 'x' is a n-dimensional tensor

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


#########################################
### Classes for variational inference ###
#########################################

# TODO: pass the correct parameters to each method
class VariationalInference(object):
    """
    Abstract class for variational inference methods.
    """
    def __init__(self, bayes):
        if not isinstance(bayes, Bayesian):
            raise TypeError(f"Model {torch.typename(bayes)} must be a Bayesian model.")
        self.bayes = bayes

    def update_bayes(self):
        raise NotImplementedError("The 'update_bayes' method must be implemented in a derived class.")

    def forward(self, input):
        raise NotImplementedError("The 'forward' method must be implemented in a derived class.")

    def train(self):
        raise NotImplementedError("The 'train' method must be implemented in a derived class.")

    def sample(self, n_samples):
        raise NotImplementedError("The 'sample' method must be implemented in a derived class.")


# TODO: minimize code duplication
class Flow(object):
    
    def __init__(self):
        pass

    def compute_flow(self, x):
        pass

    def compute_sum_log_det(self, x):
        pass


# CHECK THIS: https://github.com/kamenbliznashki/normalizing_flows/blob/master/planar_flow.py

class PlanarFlow(Flow):

    def __init__(self, x, k=1):
        super(PlanarFlow, self).__init__()

        class PlanarBlock(torch.nn.Module):
            def __init__(self, n, scaling=0.01):
                super(PlanarBlock, self).__init__()
                self.w = Parameter(scaling * torch.randn(n,))
                self.u = Parameter(scaling * torch.randn(n,))
                self.b = Parameter(torch.randn(1,).fill_(0))
            def forward(self, x):
                uTw = torch.dot(self.u, self.w)
                if uTw < -1:
                    m = lambda x: torch.log(1 + torch.exp(x)) - 1
                    self.u.data = self.u.data + ((m(uTw) - uTw) * self.w / torch.dot(self.w, self.w)).detach()
                x = x + self.u * torch.tanh(torch.dot(self.w, x) + self.b)
                return x

        self.flow = nn.Sequential(*[PlanarBlock(x.shape[0]) for _ in range(k)])

    def parameters(self):
        return self.flow.parameters()

    def compute_flow(self, x):
        return self.flow.forward(x)

    def compute_sum_log_det(self, x):
        sum_log_det = 0.
        for module in self.flow:
            uTw = torch.dot(module.u, module.w)
            if uTw < -1:
                m = lambda x: torch.log(1 + torch.exp(x)) - 1
                module.u.data = module.u.data + ((m(uTw) - uTw) * module.w / torch.dot(module.w, module.w)).detach()
            # det = torch.abs(1 + (1 - torch.tanh(torch.dot(module.w, x) + module.b) ** 2) * torch.dot(module.u, module.w))
            det = 1 + (1 - torch.tanh(torch.dot(module.w, x) + module.b) ** 2) * torch.dot(module.u, module.w)
            sum_log_det += torch.log(det)
        return sum_log_det


# class RadialFlow(Flow):

#     def __init__(self, x, k=1):
#         super(RadialFlow, self).__init__()

#         class RadialBlock(torch.nn.Module):
#             def __init__(self, n):
#                 super(RadialBlock, self).__init__()
#                 self.beta = Parameter(torch.randn(1))
#                 self.alpha = Parameter(torch.randn(1))
#                 self.z = Parameter(torch.randn(n,))
#             def forward(self, x):
#                 ???

#         self.flow = nn.Sequential(*[RadialBlock(x.shape[0]) for _ in range(k)])

#     def compute_flow(self, x):
#         return self.flow.forward(x)

#     def compute_sum_log_det(self, x):
#         sum_log_det = torch.tensor(0.0).to(device)
#         for module in self.flow:
#             det = ???
#             sum_log_det += torch.log(det)
#         return sum_log_det


class SVI(VariationalInference):
    """
    Stochastic variatiational inference based on the maximization of the ELBO with (simple) normalizing flows.
    """
    def __init__(self, bayes, n_samples=20, flow='planar', k=1):
        super(SVI, self).__init__(bayes)

        # NOTE: we assume that the network parameters are i.i.d.
        self.q0 = Gaussian() # initial distribution
        self.n_samples = n_samples
        self.n_params = self.bayes.count_parameters()

        if flow == 'planar':
            self.flow = PlanarFlow(parameters_to_vector(self.bayes.parameters(), grad=False), k)
        # elif flow == 'radial':
        #     self.flow = RadialFlow(parameters_to_vector(self.bayes.parameters(), grad=False), k)
        else:
            raise ValueError(f"Flow type {flow} is not supported.")

    def update_bayes(self):
        theta0 = parameters_to_vector(self.bayes.parameters(), grad=False)
        # print("Initial parameters:", theta0)
        theta = self.flow.compute_flow(theta0)
        # print("Parameters post flow:", theta)
        vector_to_parameters(theta, self.bayes.parameters(), grad=False)

    def compute_ELBO(self, target, output, ntrain):
        """Compute the evidence lower bound (ELBO)."""
        theta0 = self.q0.sample((self.n_samples, self.n_params)) # sample from the initial distribution
        log_joint, log_q = 0., 0.
        for idx in range(self.n_samples):
            vector_to_parameters(theta0[idx], self.bayes.parameters(), grad=False) # assign the samples to the model parameters
            log_joint += self.bayes._log_joint(target, output, ntrain) # compute the log joint with the samples
            theta0_vec = parameters_to_vector(self.bayes.parameters(), grad=False)
            log_q += self.q0.log_prob(theta0_vec) - self.flow.compute_sum_log_det(theta0_vec)
        log_joint, log_q = log_joint / self.n_samples, log_q / self.n_samples
        return log_joint - log_q

    def forward(self, input):
        return self.bayes.model.forward(input)

    def __call__(self, input):
        return self.forward(input)

    def train(self, mu, u, ntrain, epochs, optim=torch.optim.Adam, lr=0.01, loss=None, error=None, nvalid=0, batchsize=None):
        
        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise ValueError("The trainer must be set in the Bayesian model before training.")

        optimizer = optim(self.flow.parameters(), lr=lr) # optimize the variational parameters

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

                self.update_bayes() # update the Bayesian model

                optimizer.zero_grad() # zero the gradients of the optimizer

                Upred = self(*(Mtrain)) # call the ROM 'forward()' method with multiple inputs
                ELBO = self.compute_ELBO(getout(Utrain), Upred, ntrain)
                ELBO.backward() # compute gradients of ELBO

                optimizer.step() # update variational parameters

                err.append([errorf(getout(Utrain), self(*Mtrain)).item(), validerr(), testerr()])

                # Early stopping
                if(nvalid > 0 and len(err) > 3): # check if validation error is increasing
                    if((err[-1][1] > err[-2][1]) and (err[-1][0] < err[-2][0])):
                        if((err[-2][1] > err[-3][1]) and (err[-2][0] < err[-3][0])):
                            print("Early stopping.")
                            break

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {err[-1][0]:.6f}, valid: {err[-1][1]} test: {err[-1][2]:.6f}, ELBO: {ELBO.item():.6f}")
                pbar.update()

        # Update the Bayesian model
        self.update_bayes()

    def sample(self, input, n_samples):

        outputs = []
        theta0 = self.q0.sample((self.n_samples, self.n_params)) # sample from the initial distribution

        for idx in range(n_samples):
            vector_to_parameters(theta0[idx], self.bayes.parameters(), grad=False) # assign the samples to the model parameters
            outputs.append(self.forward(input))

        outputs = torch.stack(outputs)
        output_mean = torch.mean(outputs, dim=0)
        output_var = torch.var(outputs, dim=0)

        beta_inv = torch.exp(-self.bayes.log_beta).item()
        noise_var = torch.tensor(beta_inv).to(device)

        return output_mean, output_var + noise_var


# TODO: convert to class
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
    def __init__(self, bayes, n_samples=20, kernel='rbf'):
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
            if torch.cuda.is_available():
                model.cuda() # TODO: put this in a better place
            models.append(model)
        self.models = models # list of Bayesian models
        del models

    def __getitem__(self, idx):
        return self.models[idx]

    def get_particles(self):
        """Return the particles (models) of the ensemble as a dictionary."""
        particles = {}
        for idx, model in enumerate(self.models):
            particles[f"bayes.{idx}"] = model.state_dict()
        return particles
    
    def save_particles(self, path):
        """Save the particles (models) of the ensemble as a dictionary."""
        particles = self.get_particles()
        torch.save(particles, path)

    def update_bayes(self):
        theta = []
        for i in range(self.n_samples):
            vec_param = parameters_to_vector(self.models[i].parameters(), grad=False)
            theta.append(torch.unsqueeze(vec_param, 0))
        theta = torch.cat(theta)
        theta_mean = torch.mean(theta, dim=0)
        vector_to_parameters(theta_mean, self.bayes.parameters(), grad=False)

    def load_particles(self, path):
        """Load the particles (models) of the ensemble from a previous training."""
        particles = torch.load(path, map_location=device)
        for idx, model in enumerate(self.models):
            model.load_state_dict(particles[f"bayes.{idx}"])
        self.update_bayes()

    @torch.no_grad()
    def He(self, linear=False, a=0.1, seed=None):
        """He initialization.
        """
        self.bayes.He(linear=linear, a=a, seed=seed)
        for model in self.models:
            model.He(linear=linear, a=a, seed=seed)

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
    def train(self, mu, u, ntrain, epochs, optim=torch.optim.Adam, lr=0.01, lr_noise=0.01, loss=None, error=None, nvalid=0, batchsize=None):

        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise ValueError("The trainer must be set in the Bayesian model before training.")

        # TODO: support batch computation
        if batchsize is not None:
            raise NotImplementedError("Batch computation is not supported.")

        # Store n_samples instances of the optimizer
        # NOTE: LBFGS does not support per-parameter options and parameter groups
        optimizers = []
        for idx in range(self.n_samples):
            parameters = [{'params': self.models[idx].model.parameters()}, # parameters of the ROM model
                          {'params': [self.models[idx].log_beta], 'lr': lr_noise}]
            optimizer = optim(parameters, lr=lr)
            optimizers.append(optimizer)

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

                    log_posterior_i = self.models[i]._log_posterior(Upred_i, getout(Utrain), ntrain)
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
                    optimizers[i].step()
                
                err.append([errorf(getout(Utrain), self(*Mtrain)).item(), validerr(), testerr()])

                # Early stopping
                if(nvalid > 0 and len(err) > 3): # check if validation error is increasing
                    if((err[-1][1] > err[-2][1]) and (err[-1][0] < err[-2][0])):
                        if((err[-2][1] > err[-3][1]) and (err[-2][0] < err[-3][0])):
                            print("Early stopping.")
                            break

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {err[-1][0]:.6f}, valid: {err[-1][1]} test: {err[-1][2]:.6f}")
                pbar.update()

        # Update the Bayesian model
        self.update_bayes()

    @torch.no_grad()
    def sample(self, input, n_samples):
        if n_samples > self.n_samples:
            raise ValueError(f"The number of samples ({n_samples}) exceeds the number of instances ({self.n_samples}).")

        outputs = self.forward(input)
        outputs = outputs[:n_samples]

        output_mean = torch.mean(outputs, dim=0) # E[y + n] = E[y] since the additive noise has zero mean
        output_var = torch.var(outputs, dim=0)

        betas_inv = torch.tensor([torch.exp(-model.log_beta).item() for model in self.models[:n_samples]]).to(device)
        noise_var = torch.mean(betas_inv)

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
        # TODO: uncomment this (!)
        # if not isinstance(model, ROM):
        #     raise TypeError(f"Model {torch.typename(model)} must be a ROM model or one of its subclasses.")
        self.model = model

        # Prior for log precision of weights
        # TODO: set as static?
        self.alpha_a = 1. # prior shape
        self.alpha_b = 0.05 # prior rate

        # Additive noise model
        # TODO: set as static?
        self.beta_a = 2. # noise precision shape
        self.beta_b = 1e-6 # noise precision rate
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))
        self.log_beta.data = self.log_beta.to(device) # TODO: get device from ROM?

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
        self.log_beta.data = self.log_beta.to(device) # TODO: check if necessary

    @torch.no_grad()
    def Xavier(self):
        """Xavier initialization.
        """
        self.model.Xavier() # calls the ROM 'Xavier' method
        self.log_beta = Parameter(torch.log(Gamma(self.beta_a, self.beta_b).sample((1,))))
        self.log_beta.data = self.log_beta.to(device) # TODO: check if necessary

    # def cuda(self):
    #     """Move model to GPU.
    #     """
    #     self.model.cuda()
    #     self.log_beta.data = self.log_beta.to('cuda:0')

    # def cpu(self):
    #     """Move model to CPU.
    #     """
    #     self.model.cpu()
    #     self.log_beta.data = self.log_beta.to('cpu')

    def count_parameters(self):
        """Count the number of parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters()) + self.log_beta.numel()

    def forward(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'forward' method.")
        return self.trainer.forward(*args, **kwargs)
    
    def _log_joint(self, target, output, ntrain):
        """Compute the log joint."""
        log_likelihood = self.log_likelihood(target, output, self.log_beta, ntrain) # TODO: normalization constant ntrain / target.size[0]?
        # log Gamma(beta| a, b)
        log_prior_log_beta = (self.beta_a - 1.) * self.log_beta - torch.exp(self.log_beta) * self.beta_b
        return torch.sum(log_likelihood + log_prior_log_beta) # return a scalar

    def _log_prior(self):
        """Compute the log prior on the parameters (weights)."""
        # log StudentT(w | mu, lambda = a / b, nu = 2 * a)
        log_prior_w = torch.tensor(0.0).to(device) # TODO: where is device defined?
        for param in self.model.parameters():
            log_prior_w += torch.sum(torch.log1p(0.5 / self.alpha_b * param ** 2))
            log_prior_w *= - (self.alpha_a + 0.5)
        return log_prior_w

    def _log_posterior(self, target, output, ntrain):
        """Compute the UNNORMALIZED log posterior."""
        log_joint = self._log_joint(target, output, ntrain)
        log_prior = self._log_prior()
        return log_joint + log_prior

    def train(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'train' method.")
        return self.trainer.train(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.trainer is None:
            raise ValueError("A trainer must be set before calling the 'sample' method.")
        return self.trainer.sample(*args, **kwargs)

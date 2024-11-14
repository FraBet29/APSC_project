import torch
import torch.nn as nn

import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm

from __future__ import annotations
from typing import Union

from bayesian import *


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


# TODO: convert to class?
def RBF_kernel(X, h=-1):

    X_norm_squared = torch.sum(X ** 2, dim=1, keepdim=True)
    pairwise_dists_squared = X_norm_squared + X_norm_squared.T - 2 * torch.mm(X, X.T)

    if h < 0: # if h < 0, use median trick
        h = torch.median(pairwise_dists_squared)
        h = math.sqrt(0.5 * h / math.log(X.shape[0]))

    Kxx = torch.exp(-0.5 * pairwise_dists_squared / h ** 2)
    dxKxx = (torch.diag(torch.sum(Kxx, 1)) - Kxx) @ X / (h ** 2)

    return Kxx, dxKxx


class SVGD(VariationalInference):
    """
    Stein variational gradient descent.
    """
    def __init__(self, bayes, n_samples=20, kernel='rbf'):
        super(SVGD, self).__init__(bayes)

        self.n_samples = n_samples

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if kernel == 'rbf':
            self.kernel = RBF_kernel
        else:
            raise ValueError(f"Kernel type {kernel} is not supported.")

        # Store n_samples instances of the model
        models = []
        for _ in range(n_samples):
            model = deepcopy(bayes)
            # if torch.cuda.is_available():
            #     model.cuda() # NOTE: shouln't be necessary
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

    def update_bayes(self) -> None:
        theta = []
        for i in range(self.n_samples):
            vec_param = parameters_to_vector(self.models[i].parameters(), grad=False)
            theta.append(torch.unsqueeze(vec_param, 0))
        theta = torch.cat(theta)
        theta_mean = torch.mean(theta, dim=0)
        vector_to_parameters(theta_mean, self.bayes.parameters(), grad=False)

    def load_particles(self, path):
        """Load the particles (models) of the ensemble from a previous training."""
        particles = torch.load(path, map_location=self.device)
        for idx, model in enumerate(self.models):
            model.load_state_dict(particles[f"bayes.{idx}"])
        self.update_bayes()

    def He(self, linear=False, a=0.1, seed=None):
        """He initialization.
        """
        self.bayes.He(linear=linear, a=a, seed=seed)
        for model in self.models:
            model.He(linear=linear, a=a, seed=seed)

    def Hybrid(self, x1, x2):
        """Hybrid initialization.
        """
        self.bayes.Hybrid(x1, x2)
        for model in self.models:
            model.Hybrid(x1, x2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns the output of the ensemble of models.
        """
        outputs = []
        for idx in range(self.n_samples):
            outputs.append(self.models[idx].model.forward(input)) # calls the ROM 'forward()' method
        outputs = torch.stack(outputs)
        return outputs
    
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns the average output of the ensemble of models.
        """
        output = torch.zeros_like(input)
        for idx in range(self.n_samples):
            output += self.models[idx].model.forward(input) # calls the ROM 'forward()' method
        return output / self.n_samples

    # TODO: replace for loop with torch.stack?
    def train(self, mu: Union[torch.Tensor, tuple[torch.Tensor, ...]], u: Union[torch.Tensor, tuple[torch.Tensor, ...]], 
              ntrain: int, epochs: int, optim: torch.optim = torch.optim.Adam, lr: float = 0.01, lr_noise: float = 0.01, 
              loss = None, error = None, nvalid: int = 0, batchsize: int = None):

        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise RuntimeError("The trainer must be set in the Bayesian model before training.")

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

        if (M == None):
            raise RuntimeError("Input data should be either a torch.Tensor or a tuple of torch.Tensors.")
        if (U == None):
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
                    model.zero_grad()

                grad_log_posterior = [] # gradients of log posterior
                theta = [] # model parameters (particles)

                Upred = torch.zeros_like(getout(Utrain))

                for i in range(self.n_samples):

                    Upred_i = self.models[i].model.forward(*(Mtrain)) # call the ROM 'forward()' method
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
                if (nvalid > 0 and len(err) > 3): # check if validation error is increasing
                    if ((err[-1][1] > err[-2][1]) and (err[-1][0] < err[-2][0])):
                        if ((err[-2][1] > err[-3][1]) and (err[-2][0] < err[-3][0])):
                            print("Early stopping.")
                            break

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {err[-1][0]:.6f}, valid: {err[-1][1]} test: {err[-1][2]:.6f}")
                pbar.update()

        # Update the Bayesian model
        self.update_bayes()

    @torch.no_grad()
    def sample(self, input: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n_samples > self.n_samples:
            raise ValueError(f"The number of samples ({n_samples}) exceeds the number of instances ({self.n_samples}).")

        outputs = self.forward(input)
        outputs = outputs[:n_samples]

        output_mean = torch.mean(outputs, dim=0) # E[y + n] = E[y] since the additive noise has zero mean
        output_var = torch.var(outputs, dim=0)

        betas_inv = torch.tensor([torch.exp(-model.log_beta).item() for model in self.models[:n_samples]]).to(self.device)
        noise_var = torch.mean(betas_inv)

        return output_mean, output_var + noise_var
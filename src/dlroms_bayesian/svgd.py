from __future__ import annotations
from typing import *

import torch
import torch.nn as nn

import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm

from dlroms_bayesian.bayesian import Bayesian
from dlroms_bayesian.bayesian import VariationalInference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parameters_to_vector(params: Iterator[nn.Parameter], grad: bool = True) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert parameters (and their gradients) to a tensor.
    Args:
        params (Iterator[nn.Parameter]): iterator of parameters (as given by the 'parameters' method in PyTorch)
        grad (bool): whether to also convert gradients
    Returns:
        torch.Tensor: vector of parameters
        torch.Tensor: vector of gradients (if grad=True)
    """
    if grad:
        vec_params, vec_grads = [], []
        for param in params:
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)
    else:
        vec_params = [param.data.view(-1) for param in params]
        return torch.cat(vec_params)


def vector_to_parameters(vec: torch.Tensor, params: Iterator[nn.Parameter], grad: bool = True) -> None:
    """
    Convert a tensor to parameters or gradients of the parameters.
    Args:
        vec (torch.Tensor): vector of parameters or gradients
        params (Iterator[nn.Parameter]): iterator of parameters (as given by the 'parameters' method in PyTorch)
        grad (bool): whether to convert gradients
    Returns:
        None
    """
    idx = 0
    for param in params:
        n_param = torch.prod(torch.LongTensor(list(param.size())))
        if grad:
            param.grad.data = vec[idx:idx+n_param].view(param.size())
        else:
            param.data = vec[idx:idx+n_param].view(param.size())
        idx += n_param


def rbf_kernel(X: torch.Tensor, h: float = -1.) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Radial basis function kernel.
    Args:
        X (torch.Tensor): input
        h (float): bandwidth (default: -1)
    Returns:
        tuple[torch.Tensor, torch.Tensor]: the kernel and its gradient
    """
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
    Stein variational gradient descent (SVGD) algorithm.
    """
    def __init__(self, bayes: Bayesian, n_samples: int = 20, kernel: str = 'rbf'):
        """
        Initialize the SVGD algorithm.
        Args:
            bayes (Bayesian): Bayesian model
            n_samples (int): number of samples (default: 20)
            kernel (str): kernel type (default: 'rbf')
        """
        super(SVGD, self).__init__(bayes)

        self.n_samples = n_samples
        
        if kernel == 'rbf':
            self.kernel = rbf_kernel
        else:
            raise ValueError(f"Kernel type {kernel} is not supported.")

        # Store n_samples instances of the model
        models = []
        for _ in range(n_samples):
            model = deepcopy(bayes)
            models.append(model)
        self.models = models # list of Bayesian models

    def __getitem__(self, idx):
        return self.models[idx]

    def get_particles(self) -> dict:
        """
        Return the particles (models) of the ensemble as a dictionary.
        Args:
            None
        Returns:
            dict: particles
        """
        particles = {}
        for idx, model in enumerate(self.models):
            particles[f"bayes.{idx}"] = model.state_dict()
        return particles
    
    def load_particles(self, path: str) -> None:
        """
        Load the particles (models) of the ensemble from a checkpoint.
        Args:
            path (str): path to the checkpoint
        Returns:
            None
        """
        particles = torch.load(path, weights_only=True, map_location=device)
        for idx, model in enumerate(self.models):
            model.load_state_dict(particles[f"bayes.{idx}"])
        self.update()
    
    def save_particles(self, path: str) -> None:
        """
        Save the particles (models) of the ensemble as a dictionary.
        Args:
            path (str): path to the checkpoint
        Returns:
            None
        """
        particles = self.get_particles()
        torch.save(particles, path)

    def update(self) -> None:
        """
        Update the Bayesian model with the average of the ensemble.
        """
        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise RuntimeError("The trainer must be set in the Bayesian model before updating the model.")

        theta = []
        for i in range(self.n_samples):
            vec_param = parameters_to_vector(self.models[i].parameters(), grad=False)
            theta.append(torch.unsqueeze(vec_param, 0))
        theta = torch.cat(theta)
        theta_mean = torch.mean(theta, dim=0)
        vector_to_parameters(theta_mean, self.bayes.parameters(), grad=False)

    def He(self, linear=False, a=0.1, seed=None):
        """
        He initialization (see DLROMs Layer class).
        """
        self.bayes.He(linear=linear, a=a, seed=seed)
        for model in self.models:
            model.He(linear=linear, a=a, seed=seed)

    def hybrid(self):
        """
        Hybrid initialization.
        """
        self.bayes.hybrid()
        for model in self.models:
            model.hybrid()

    def forward(self, input: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """
        Compute a forward pass through the ensemble of models.
        Args:
            input (torch.Tensor): input data
            reduce (bool): whether to return the mean of the ensemble or the full output (default: True)
        Returns:
            torch.Tensor: output tensor (either the mean output or the ensemble of outputs)
        """
        outputs = [bayes.model.forward(input) for bayes in self.models] # calls the ROM 'forward' method
        outputs = torch.stack(outputs)
        return torch.mean(outputs, dim=0) if reduce else outputs

    def __call__(self, input: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """
        Calls the 'forward' method.
        """
        return self.forward(input, reduce=reduce)

    def train(self, mu: Union[torch.Tensor, tuple[torch.Tensor, ...]], u: Union[torch.Tensor, tuple[torch.Tensor, ...]], 
              ntrain: int, epochs: int, loss: Callable, optim: torch.optim.Optimizer = torch.optim.Adam, lr: float = 0.01, lr_noise: float = 0.01, 
              error: Optional[Callable] = None, nvalid: int = 0, adaptive: bool = False, track_history: bool = False) -> Optional[dict]:
        """
        Train the ensemble of models using SVGD.
        Args:
            mu (torch.Tensor or tuple[torch.Tensor, ...]): input data
            u (torch.Tensor or tuple[torch.Tensor, ...]): output data
            ntrain (int): number of training samples
            epochs (int): number of training epochs
            loss (Callable): loss function
            optim (torch.optim.Optimizer): optimizer (default: Adam)
            lr (float): learning rate (default: 0.01)
            lr_noise (float): learning rate for the noise (default: 0.01)
            error (Optional[Callable]): error function (default: None, in which case the loss function is used)
            nvalid (int): number of validation samples (default: 0)
            adaptive (bool): whether to use a ReduceLROnPlateau scheduler (default: False)
            track_history (bool): whether to track the training history (default: False)
        Returns:
            Optional[dict]: training history (if track_history = True)
        """
        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise RuntimeError("The trainer must be set in the Bayesian model before training.")

        # Store n_samples instances of the optimizer
        # NOTE: LBFGS does not support per-parameter options and parameter groups
        optimizers = []
        for idx in range(self.n_samples):
            parameters = [{'params': self.models[idx].model.parameters()}, # parameters of the ROM model
                          {'params': [self.models[idx].log_beta], 'lr': lr_noise}]
            optimizer = optim(parameters, lr=lr)
            optimizers.append(optimizer)
        
        if adaptive:
            schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=1e-6) for optimizer in optimizers]

        if track_history:
            history = {
                'err': [],
                'log_posterior': [],
                'grad_theta': []
            }

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
        trainerr = lambda: errorf(getout(Utrain), self(*Mtrain)).item()
        validerr = (lambda: np.nan) if nvalid == 0 else (lambda: errorf(getout(Uvalid), self(*Mvalid)).item())
        testerr = (lambda: np.nan) if ntest == 0 else (lambda: errorf(getout(Utest), self(*Mtest)).item())

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):

                for model in self.models:
                    model.zero_grad()

                log_posterior = 0.

                for i in range(self.n_samples):
                    Upred_i = self.models[i].model.forward(*(Mtrain)) # call the ROM 'forward' method
                    log_posterior_i = self.models[i]._log_posterior(Upred_i, getout(Utrain), ntrain) # compute log posterior
                    log_posterior += log_posterior_i
                    
                log_posterior.backward() # compute gradients of log posterior (in one backward pass)

                theta = [] # model parameters (particles)
                grad_log_posterior = [] # gradients of log posterior

                for i in range(self.n_samples):
                    vec_param, vec_grad_log_posterior = parameters_to_vector(self.models[i].parameters(), grad=True)
                    grad_log_posterior.append(vec_grad_log_posterior)
                    theta.append(vec_param)

                theta = torch.stack(theta)
                grad_log_posterior = torch.stack(grad_log_posterior)

                Kxx, dxKxx = self.kernel(theta) # compute kernel and its gradient
                grad_theta = - (torch.mm(Kxx, grad_log_posterior) + dxKxx) / self.n_samples # SVGD update
            
                for i in range(self.n_samples):
                    vector_to_parameters(grad_theta[i], self.models[i].parameters(), grad=True)
                    optimizers[i].step() # update parameters

                err = (trainerr(), validerr(), testerr())

                if adaptive:
                    metric = err[1] if nvalid > 0 else err[0]
                    for scheduler in schedulers:
                        scheduler.step(metric) # update learning rate

                if track_history:
                    history['err'].append(err)
                    history['log_posterior'].append(log_posterior.item())
                    history['grad_theta'].append(torch.linalg.norm(grad_theta).item())

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, train: {err[0]:.3e}, valid: {err[1]:.3e}, test: {err[2]:.3e}")
                pbar.update()

        # Update the Bayesian model
        self.update()

        if track_history:
            return history

    @torch.no_grad()
    def sample(self, input: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the ensemble of models.
        Args:
            input (torch.Tensor): input tensor
            n_samples (int): number of samples
        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and variance of the samples
        """
        if not (self is self.bayes.trainer): # avoid calling a trainer different from the one set in the Bayesian model
            raise RuntimeError("The trainer must be set in the Bayesian model before sampling.")

        if n_samples > self.n_samples:
            raise ValueError(f"The number of samples ({n_samples}) exceeds the number of instances ({self.n_samples}).")

        outputs = self(input, reduce=False)
        outputs = outputs[:n_samples]

        output_mean = torch.mean(outputs, dim=0) # E[y+n] = E[y] since the n has zero mean
        output_var = torch.var(outputs, dim=0) # Var[y+n] = Var[y] + Var[n] since y and n are independent

        betas_inv = torch.tensor([torch.exp(-model.log_beta).item() for model in self.models[:n_samples]]).to(device)
        noise_var = torch.mean(betas_inv)

        return output_mean, output_var + noise_var
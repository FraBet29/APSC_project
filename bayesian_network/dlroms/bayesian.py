import torch
from copy import deepcopy
import math
from time import time
from tqdm import tqdm

from minns import *
from roms import *

from torch.distributions import Gamma
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
# plt.switch_backend('agg')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parameters_to_vector(parameters, grad=False, both=False):
    """Convert parameters or/and their gradients to one vector
    Arguments:
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): Vectorizes gradients if true, otherwise vectorizes params
        both (bool): If True, vectorizes both parameters and their gradients,
            `grad` has no effect in this case. Otherwise vectorizes parameters
            or gradients according to `grad`.
    Returns:
        The parameters or/and their gradients (each) represented by a single
        vector (th.Tensor, not Variable)
    """
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                vec.append(param.data.view(-1))
        else:
            for param in parameters:
                vec.append(param.grad.data.view(-1))
        return torch.cat(vec)
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
    # Ensure vec of type Variable
    if not isinstance(vec, (torch.FloatTensor, torch.cuda.FloatTensor)):
        raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size())
            # Increment the pointer
            pointer += num_param
    else:
        for param in parameters:
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[pointer:pointer + num_param].view(param.size())
            # Increment the pointer
            pointer += num_param


class Bayesian(torch.nn.Module):
    """Class for Bayesian NNs with Stein Variational Gradient Descent.
    
    Bayesian NNs: y = f(x, w) + n

	### TODO: change this using the prior from the mesh-informed layer ###
    uncertain weights:
            w_i ~ Normal(w_i | mu=0, 1 / alpha)
            alpha ~ Gamma(alpha | shape=1, rate=0.05) (shared)
            --> w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
            Parameterization of StudentT in Bishop p.103 Eq. (2.159)

    Assumptions on noise:
        Additive, Gaussian, homoscedastic (independent of input), 
        output wise (same for every pixels in the output).
            n ~ Normal(0, 1 / beta)
            beta ~ Gamma(beta | shape=2, rate=2e-6)

    Hyperparameters for weights and noise are pre-defined based on heuristic.

    Given a deterministic `model`, initialize `n_samples` replicates
    of the `model`. (plus `n_samples` of noise precision realizations)

    `model` must implement `reset_parameters` method for the replicates
    to have different initial realizations of uncertain parameters.

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The deterministic NN to be instantiated `n_samples` 
            times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """
    def __init__(self, model, n_samples=20):
        super(Bayesian, self).__init__()
        if not isinstance(model, ROM):
            raise TypeError("model {} is not a ROM subclass".format(torch.typename(model)))

        self.n_samples = n_samples

        # TODO: replace prior on the weights with mesh-informed layer
        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # for efficiency, represent StudentT params using Gamma params
        self.w_prior_shape = 1.
        self.w_prior_rate = 0.05
        
        # TODO: decide how to model noise
        # noise variance 1e-6: beta ~ Gamma(beta | shape, rate)
        self.beta_prior_shape = 2.
        self.beta_prior_rate = 1.e-6

        # replicate `n_samples` instances with the same network as `model`
        instances = []
        for i in range(n_samples):
            new_instance = deepcopy(model)
            # initialize each model instance with their defualt initialization
            new_instance.He() # He initialization
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
        # self.nnets = torch.nn.ModuleList(instances) # TODO: check if compatible with type of mesh-informed layer
        self.nnets = instances # list of ROMs
        del instances

        # log precision (Gamma) of Gaussian noise
        log_beta = Gamma(self.beta_prior_shape, 
                         self.beta_prior_rate).sample((self.n_samples,)).log()
        for i in range(n_samples):
            self.nnets[i].log_beta = Parameter(log_beta[i]) # added automatically

        print('Total number of parameters: {}'.format(self.num_parameters()))

    def num_parameters(self):
        count = 0
        # for name, param in self.named_parameters(): # does not see all the layers of a ROM
        #     print(name, param.shape, param.numel())
        #     count += param.numel()
        for model in self.nnets:
            for param in model.parameters():
                count += param.numel() # does not count 'log_beta'
                count += 1 # add 'log_beta'
        return count

    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta.item() for i in range(self.n_samples)], device=device)

    def forward(self, input):
        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(input)) # calls the ROM 'forward()' method
        output = torch.stack(output)

        return output

    def log_joint(self, index, output, target, ntrain):
        """Log joint probability or unnormalized posterior for single model
        instance. Ignoring constant terms for efficiency.
        Can be implemented in batch computation, but memory is the bottleneck.
        Thus here we trade computation for memory, e.g. using for loop.

        Args:
            index (int): model index, 0, 1, ..., `n_samples`
            output (Tensor)
            target (Tensor) 
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob

        Returns:
            Log joint probability (zero-dim tensor)
        """
        # Normal(target | output, 1 / beta * I)
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * (target - output).pow(2).sum()
                            + 0.5 * target.numel() * self.nnets[index].log_beta)
        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.0).to(device)
        for param in self.nnets[index].parameters():
            log_prob_prior_w += \
                torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        # log prob of prior of log noise-precision (NOT noise precision)
        log_prob_prior_log_beta = ((self.beta_prior_shape-1.0) * self.nnets[index].log_beta \
                    - self.nnets[index].log_beta.exp() * self.beta_prior_rate)
        return log_likelihood + log_prob_prior_w + log_prob_prior_log_beta

    def predict(self, x_test):
        """
        Predictive mean and variance at x_test.
        Args:
            x_test (Tensor): [N, *], test input
        """
        y = self.forward(x_test) # the 'forward' method stacks the outputs from the different model instances
        y_pred_mean = y.mean(0) # mean over the different instances

        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (- self.log_beta).exp()
        y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y_pred_mean, y_pred_var


class SVGD(object):
    """Base class for Stein Variational Gradient Descent, with for-loops...
    TODO: implement with batch computation

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn, n_samples=20, lr=0.01, lr_noise=0.01):
        
        self.bayes_nn = bayes_nn
        self.n_samples = n_samples

        optimizers = []
        schedulers = []

        for i in range(self.n_samples):
            parameters = [{'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise},
                    {'params': self.bayes_nn[i].parameters()}]
            optimizer = torch.optim.Adam(parameters, lr=lr) # TODO: optimizer choice?
            optimizers.append(optimizer)
            schedulers.append(ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)) # TODO: scheduler choice?
        
        self.optimizers = optimizers
        self.schedulers = schedulers
        del optimizers, schedulers

    def squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2

        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of 
                one sample

        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)

    def Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self.squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx

    def train(self, mu, u, ntrain, epochs, nvalid=0):

        # M = (mu,) if(isinstance(mu, torch.Tensor)) else (mu if (isinstance(mu, tuple)) else None) # see ROM... TODO: why though?
        # U = (u,) if(isinstance(u, torch.Tensor)) else (u if (isinstance(u, tuple)) else None)
        M = mu
        U = u

        # if(M == None):
        #         raise RuntimeError("Input data should be either a torch.Tensor or a tuple of torch.Tensors.")
        # if(U == None):
        #         raise RuntimeError("Output data should be either a torch.Tensor or a tuple of torch.Tensors.")

        ntest = len(U) - ntrain
        # Mtrain, Utrain = tuple([m[:(ntrain-nvalid)] for m in M]), tuple([um[:(ntrain-nvalid)] for um in U])
        # Mvalid, Uvalid = tuple([m[(ntrain-nvalid):ntrain] for m in M]), tuple([um[(ntrain-nvalid):ntrain] for um in U])
        # Mtest, Utest = tuple([m[-ntest:] for m in M]), tuple([um[-ntest:]for um in U])
        Mtrain, Utrain = M[:(ntrain-nvalid)], U[:(ntrain-nvalid)]
        Mvalid, Uvalid = M[(ntrain-nvalid):ntrain], U[(ntrain-nvalid):ntrain]
        Mtest, Utest = M[-ntest:], U[-ntest:]

        # getout = (lambda y: y[0]) if len(U)==1 else (lambda y: y)

        with tqdm(total=epochs) as pbar:

            for epoch in range(epochs):

                mse = 0.0
                mse_check = 0.0

                for model in self.bayes_nn.nnets:
                    model.zero_grad() # TODO: check if all gradients are zeroed

                # all gradients of log joint probability: (S, P)
                grad_log_joint = []
                # all model parameters (particles): (S, P)
                theta = []

                # Upred = tuple([torch.zeros_like(u) for u in Utrain])
                Upred = torch.zeros_like(Utrain)

                for i in range(self.n_samples):
                    
                    Upred_i = self.bayes_nn[i].forward(Mtrain) # call the ROM 'forward()' method
                    Upred += Upred_i.detach()

                    log_joint_i = self.bayes_nn.log_joint(i, Upred_i, Utrain, ntrain)
                    
                    # compute gradients of log joint probabilities
                    log_joint_i.backward() # backward frees memory for computation graph

                    # extract parameters and their gradients out from models
                    vec_param, vec_grad_log_joint = parameters_to_vector(self.bayes_nn[i].parameters(), both=True)
                    grad_log_joint.append(vec_grad_log_joint.unsqueeze(0)) # concatenate log joint gradients
                    theta.append(vec_param.unsqueeze(0)) # concatenate parameters

                ### SVGD update ###
                # calculating the kernel matrix and its gradients
                theta = torch.cat(theta)
                Kxx, dxKxx = self.Kxx_dxKxx(theta)
                grad_log_joint = torch.cat(grad_log_joint)
                # this line needs S x P memory... TODO: check if this is true
                grad_logp = torch.mm(Kxx, grad_log_joint)
                grad_theta = - (grad_logp + dxKxx) / self.n_samples
                # explicitly deleting variables does not release memory :( TODO: check if this is true
            
                # update param gradients
                for i in range(self.n_samples):
                    vector_to_parameters(grad_theta[i], self.bayes_nn[i].parameters(), grad=True)
                    self.optimizers[i].step()
                # WEAK: no loss function to suggest when to stop or approximation performance
                mse += F.mse_loss(Upred / self.n_samples, Utrain).item()
                rmse_train = np.sqrt(mse / self.n_samples)

                for i in range(self.n_samples):
                    self.schedulers[i].step(rmse_train)

                # compute test (validation?) error
                if ntest > 0:
                    Ucheck = torch.zeros_like(Utest)

                    for i in range(self.n_samples):
                        Ucheck_i = self.bayes_nn[i].forward(Mtest)
                        Ucheck += Ucheck_i.detach()
                    
                    mse_check += F.mse_loss(Ucheck / self.n_samples, Utest).item()
                    rmse_test = np.sqrt(mse_check / self.n_samples)

                # print('Epoch: {}/{}, RMSE: {:.6f}'.format(epoch+1, epochs, rmse_train))
                if ntest > 0:
                    pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, RMSE: {rmse_train:.6f}, RMSE test: {rmse_test:.6f}")
                else:
                    pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, RMSE: {rmse_train:.6f}, RMSE test: nan")
                pbar.update(1)

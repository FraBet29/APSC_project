import numpy as np
import torch

from dlroms.dnns import Weightless, Sparse
from dlroms.fespaces import coordinates, mesh
from dlroms.minns import Local, Geodesic, Navigator
from dlroms import dnns


class Channel(Weightless):
    """
    Layer to select a channel from a 3D input tensor of shape (batch_size, channels, Nh).
    """
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
    
    def forward(self, x):
        return x[:, self.ch]


class ExpandedSparse(Sparse):
    """
    Expansion of the Sparse layer in DLROMs with new initialization methods.
    """
    def __init__(self, dists, mask, activation = dnns.leakyReLU):
        """
        Layer initialization (see DLROMs Sparse).
        """
        super(ExpandedSparse, self).__init__(mask, activation)
        self.dists = dists # store distance for active weights

    def deterministic(self):
        """
        Initializes the weights of the layer in a deterministic way, based on the distance matrix.
        """
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc[0], self.loc[1]] = self.core.tensor(np.exp(-self.dists)) # w = exp{-d}
        W = W / torch.sum(W, dim=0) # normalize
        self.weight = torch.nn.Parameter(self.core.tensor(W[self.loc]))

    def hybrid(self):
        """
        Initializes the weights of the layer using the deterministic method with a random perturbation.
        """
        self.deterministic()
        nonzeros = len(self.loc[0]) # number of active weights
        eta = self.core.tensor(np.random.randn(nonzeros))
        self.weight = torch.nn.Parameter(eta * self.weight)


class ExpandedLocal(ExpandedSparse):
    """
    Expansion of the Local layer in DLROMs with new initialization methods.
    """
    def __init__(self, x1, x2, support, activation = dnns.leakyReLU):
        """
        Layer initialization (see DLROMs Local).
        """
        from dlroms.fespaces import coordinates
        coordinates1 = x1 if(isinstance(x1, np.ndarray)) else coordinates(x1)
        coordinates2 = x2 if(isinstance(x2, np.ndarray)) else coordinates(x2)
        
        M = 0
        dim = len(coordinates1[0])
        for j in range(dim):
            dj = coordinates1[:, j].reshape(-1, 1) - coordinates2[:, j].reshape(1, -1)
            M = M + dj ** 2
        M = np.sqrt(M) # Euclidean distance matrix
        mask = M < support
        dists = M[mask]

        super(ExpandedLocal, self).__init__(dists, mask, activation)


class ExpandedGeodesic(ExpandedSparse):
    """
    Expansion of the Geodesic layer in DLROMs with new initialization methods.
    """
    def __init__(self, domain, x1, x2, support, accuracy = 1, activation = dnns.leakyReLU):
        """
        Layer initialization (see DLROMs Geodesic).
        """
        from dlroms.fespaces import coordinates, mesh
        coordinates1 = x1 if(isinstance(x1, np.ndarray)) else coordinates(x1)
        coordinates2 = x2 if(isinstance(x2, np.ndarray)) else coordinates(x2)
        try:
            navigator = Navigator(domain, mesh(domain, resolution = accuracy))
        except:
            navigator = Navigator(domain, mesh(domain, stepsize = accuracy))
        
        E1 = navigator.finde(coordinates1).reshape(-1,1)
        E2 = navigator.finde(coordinates2).reshape(1,-1)
        M = navigator.D[E1, E2] # geodesic distance matrix
        mask = M < support
        dists = M[mask]

        super(ExpandedGeodesic, self).__init__(dists, mask, activation)

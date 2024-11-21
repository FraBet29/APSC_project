import numpy
import torch

from dlroms.dnns import Weightless, Sparse
from dlroms.fespaces import coordinates
from dlroms.minns import Local, Geodesic


class Channel(Weightless):
    """
    Selects a channel from the input tensor.
    """
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
    
    def forward(self, x):
        return x[:, self.ch]


class ExpandedSparse(Sparse):
    """
    Sparse layer with new initialization methods.
    """

    def __init__(self, *args, **kwargs):
        super(ExpandedSparse, self).__init__(*args, **kwargs)

    def deterministic(self, x1, x2):
        """Initializes the weights of the layer in a deterministic way, based on the distance between the mesh nodes.
        """
        coordinates1 = x1 if(isinstance(x1, numpy.ndarray)) else coordinates(x1)
        coordinates2 = x2 if(isinstance(x2, numpy.ndarray)) else coordinates(x2)

        M = 0
        dim = len(coordinates1[0])
        for j in range(dim):
            dj = coordinates1[:, j].reshape(-1, 1) - coordinates2[:, j].reshape(1, -1)
            M = M + dj ** 2
        M = numpy.sqrt(M) # distance matrix

        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc[0], self.loc[1]] = self.core.tensor(numpy.exp(-M[self.loc[0], self.loc[1]])) # w = exp{-d}
        W = W / torch.sum(W, dim=0) # normalize

        self.weight = torch.nn.Parameter(self.core.tensor(W[self.loc]))

    def hybrid(self, x1, x2):
        """Initializes the weights of the layer in a hybrid way, based on the distance between the mesh nodes and random values.
        """
        self.deterministic(x1, x2)
        nonzeros = len(self.loc[0]) # number of active weights
        eta = self.core.tensor(numpy.random.randn(nonzeros))
        self.weight = torch.nn.Parameter(eta * self.weight)


class ExpandedLocal(Local, ExpandedSparse):
    """
    Local layer with new initialization methods.
    """
    def __init__(self, *args, **kwargs):
        super(ExpandedLocal, self).__init__(*args, **kwargs)


class ExpandedGeodesic(Geodesic, ExpandedSparse):
    """
    Geodesic layer with new initialization methods.
    """
    def __init__(self, *args, **kwargs):
        super(ExpandedGeodesic, self).__init__(*args, **kwargs)

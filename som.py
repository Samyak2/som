"""
Much of the code is modified from:
- https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

import torch
import torch.nn as nn
import numpy as np

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m*n, dim)
        # self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        xx, yy = np.meshgrid(np.linspace(0, m-1, m), np.linspace(0, n-1, n))
        self.locations = torch.LongTensor(np.array((xx.ravel(), yy.ravel())).T)
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min(range(len(self.weights)),
                            key=lambda x, vect=vect: np.linalg.norm(vect-self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def get_bmu_loc(self, x):
        dists = self.pdist(torch.stack([x for i in range(self.m*self.n)]), self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()
        return bmu_loc

    def forward(self, x, it):
        bmu_loc = self.get_bmu_loc(x)

        learning_rate_op = 1.0 - it/self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        # dist^2 = (x1-x2)^2 + (y1-y2)^2 + ...
        bmu_distance_squares = torch.sum(torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m*self.n)]).float(), 2), 1)

        # theta = exp^(-dist^2 / sigma^2)
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.m*self.n)])
        # dW = LR * (X - W)
        delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.m*self.n)]) - self.weights))
        # W = W + dW
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights


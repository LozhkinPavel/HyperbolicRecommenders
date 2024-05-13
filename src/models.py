import numpy as np
from scipy.sparse.linalg import svds

import torch
import geoopt
import torch.nn as nn
import torch.nn.init as init
from .geoopt_plusplus.modules import PoincareLinear, UnidirectionalPoincareMLR
from .geoopt_plusplus.manifolds import PoincareBall

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class PureSVD:
    def __init__(self, rank=10, randomized=False):
        self.randomized = randomized
        self.rank = rank
        self.item_factors = None
        self.train_matrix = None
        self.name = 'PureSVD'

    def fit(self, matrix):
        self.train_matrix = matrix
        *_, vt = svds(self.train_matrix, k=self.rank, return_singular_vectors='vh')
        self.item_factors = torch.tensor(np.ascontiguousarray(vt[::-1, :].T), dtype=torch.float64).cuda()

    def __call__(self, batch, *, rank=None):
        factors = self.item_factors
        if rank is not None:
            assert rank <= self.item_factors.shape[1]
            factors = self.item_factors[:, :rank]
        
        if batch.ndim == 1:
            return (factors @ (factors.T @ batch.view(-1, 1))).squeeze()
        return (batch @ factors) @ factors.T

    def get_embedding(self, batch):
        return batch @ self.item_factors

    def train(self):
        pass
    
    def eval(self):
        pass

class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(dtype=torch.float64, *args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output


class ExpMap0(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.expmap0(x)

class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, ball, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float64))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        mv = self.ball.mobius_matvec(self.weight, x)
        if self.bias is not None:
            bias = self.ball.expmap0(self.bias)
            mv = self.ball.mobius_add(mv, bias)
        return self.ball.projx(mv)


class EucTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, 1, size=(in_dim, out_dim), dtype=torch.float64), requires_grad=False)

    def forward(self, x):
        return x @ self.W



class EdgeEncoder(nn.Module):
    def __init__(self, num_items, num_users, latent_dim):
        super().__init__()
        self.num_users = num_users
        self.euc_tf = EucTransform(num_items, num_items)
        self.ind_tf = EucTransform(num_users, num_items)

    def forward(self, input):
        ind = torch.tensor(torch.nn.functional.one_hot(input[1], num_classes=self.num_users), dtype=torch.float64)
        x = self.euc_tf(input[0])
        ind = self.ind_tf(ind)
        return x + ind

class EdgeAutoEncoder(nn.Module):
    def __init__(self, num_items, num_users, latent_dim=64, c=0.5, bias=True):
        super().__init__()
        self.ball = PoincareBall(c)
        self.edge_enc = EdgeEncoder(num_items, num_users, latent_dim)
        self.encode = PoincareLinear(num_items, latent_dim, bias=bias, ball=self.ball)
        self.decode = UnidirectionalPoincareMLR(latent_dim, num_items, bias=bias, ball=self.ball)
        self.exp_map = ExpMap0(self.ball)

    def forward(self, input):
        x = self.edge_enc(input)
        x = self.encode(self.exp_map(x))
        return self.decode(x)

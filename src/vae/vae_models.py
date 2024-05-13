from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import List, Tuple

from src.vae.components.component import EuclidianComponent
from src.vae.sampling import WrappedNormalProcedure, EuclideanNormalProcedure
from src.vae.components import HyperbolicComponent
from src.vae.ops.hyperbolics import lorentz_to_poincare
from torch.distributions import Distribution



class Reparametrized:
    def __init__(self, q_z: Distribution, p_z: Distribution, z: Tensor, data: Tuple[Tensor, ...]) -> None:
        self.q_z = q_z
        self.p_z = p_z
        self.z = z
        self.data = data


Outputs = Tuple[List[Reparametrized], Tensor, Tensor]


class HyperbolicVAE(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, ball, encoder, decoder, scalar_parametrization: bool, dropout=0.2):
        super().__init__()

        self.ball = ball
        self.component = HyperbolicComponent(in_dim, emb_dim, WrappedNormalProcedure, encoder, ball)
        self.component.init_layers(scalar_parametrization)
        self.reconstruction_loss = nn.CrossEntropyLoss(reduction="none")
        self.reconstruction_loss1 = nn.BCEWithLogitsLoss(reduction='none')
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.decoder = decoder(emb_dim, in_dim, bias=True, ball=ball)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        # x = F.normalize(x)
        # x = self.drop(x)
        return x

    def decode(self, concat_z: Tensor) -> Tensor:
        return self.decoder(concat_z)

    def forward(self, x: Tensor):
        x_encoded = self.encode(x)

        q_z, p_z, _ = self.component(x_encoded)
        z, data = q_z.rsample_with_parts()
        x_ = self.decode(lorentz_to_poincare(z, self.ball.radius))
        return Reparametrized(q_z, p_z, z, data), z, x_

    def train_step(self, optimizer, x_mb: Tensor, beta: float):
        optimizer.zero_grad()

        reparametrized, concat_z, x_mb_ = self(x_mb)
        assert x_mb_.shape == x_mb.shape
        ce = self.reconstruction_loss(x_mb_, x_mb)
        bce = self.reconstruction_loss1(x_mb_, x_mb).sum(dim=-1)
        kl_comp = self.component.kl_loss(reparametrized.q_z, reparametrized.p_z, reparametrized.z, reparametrized.data)
        loss = (ce + beta * kl_comp).mean()
        assert torch.isfinite(loss).all()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        optimizer.step()

        return loss, (reparametrized, concat_z, x_mb_)


class SampleLayer(torch.nn.Module):
    def __init__(self, ball):
        super().__init__()
        self.ball = ball

    def forward(self, x):
        z, data = x[0].rsample_with_parts()
        return lorentz_to_poincare(z, self.ball.radius)


class VAE(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, scalar_parametrization: bool, dropout=0.2):
        super().__init__()

        self.component = EuclidianComponent(in_dim, emb_dim, EuclideanNormalProcedure)
        self.component.init_layers(scalar_parametrization)
        self.reconstruction_loss = nn.CrossEntropyLoss(reduction="none")
        self.reconstruction_loss1 = nn.BCEWithLogitsLoss(reduction='none')
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.decoder = nn.Linear(emb_dim, in_dim, bias=True, dtype=torch.float64)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        # x = F.normalize(x)
        # x = self.drop(x)
        return x

    def decode(self, concat_z: Tensor) -> Tensor:
        return self.decoder(concat_z)

    def forward(self, x: Tensor):
        x_encoded = self.encode(x)

        q_z, p_z, _ = self.component(x_encoded)
        z, data = q_z.rsample_with_parts()
        x_ = self.decode(z)
        return Reparametrized(q_z, p_z, z, data), z, x_

    def train_step(self, optimizer, x_mb: Tensor, beta: float):
        optimizer.zero_grad()

        reparametrized, concat_z, x_mb_ = self(x_mb)
        assert x_mb_.shape == x_mb.shape
        ce = self.reconstruction_loss(x_mb_, x_mb)
        bce = self.reconstruction_loss1(x_mb_, x_mb).sum(dim=-1)
        kl_comp = self.component.kl_loss(reparametrized.q_z, reparametrized.p_z, reparametrized.z, reparametrized.data)
        loss = (bce + beta * kl_comp).mean()
        assert torch.isfinite(loss).all()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        optimizer.step()

        return loss, (reparametrized, concat_z, x_mb_)

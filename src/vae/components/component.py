# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, Tuple, TypeVar, Type, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn.functional as F
import numpy as np

from ..ops import Manifold, Hyperboloid, Euclidean
from ..ops.poincare import poincare_to_lorentz
from ..sampling import SamplingProcedure
from ...batchmodels import ExpMap0

Q = TypeVar('Q', bound=Distribution)
P = TypeVar('P', bound=Distribution)


class HyperbolicComponent(torch.nn.Module):

    def forward(self, x: Tensor) -> Tuple[Q, P, Tuple[Tensor, ...]]:
        z_params = self.encode(x)
        q_z, p_z = self.reparametrize(*z_params)
        return q_z, p_z, z_params

    def __init__(self, in_dim, out_dim, sampling_procedure: Type[SamplingProcedure[Q, P]], fc_mean_factory, ball) -> None:
        super().__init__()
        self.radius = ball.radius
        self.ball = ball
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._sampling_procedure_type = sampling_procedure
        self.sampling_procedure: SamplingProcedure[Q, P] = None
        self.manifold: Manifold = None
        self.fc_mean_factory = fc_mean_factory

        self.fc_mean = None
        self.fc_logvar = None

    def init_layers(self, scalar_parametrization: bool) -> None:
        self.manifold = self.create_manifold()
        self.sampling_procedure = self._sampling_procedure_type(self.manifold, scalar_parametrization)

        self.fc_mean = torch.nn.Sequential(ExpMap0(self.ball), self.fc_mean_factory(self.in_dim, self.out_dim,
                                                                                    bias=True, ball=self.ball))

        if scalar_parametrization:
            self.fc_logvar = torch.nn.Linear(self.in_dim, 1, dtype=torch.float64)
        else:
            self.fc_logvar = torch.nn.Linear(self.in_dim, self.out_dim, dtype=torch.float64)
            

    @property
    def device(self) -> torch.device:
        return self.fc_logvar.weight.device

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_mean = self.fc_mean(x)
        assert torch.isfinite(z_mean).all()
        z_mean_h = poincare_to_lorentz(z_mean, self.manifold.radius)
        assert torch.isfinite(z_mean_h).all()

        z_logvar = self.fc_logvar(x)
        assert torch.isfinite(z_logvar).all()
        # +eps prevents collapse
        if self.training:
            std = F.softplus(z_logvar) + 1e-5
        else:
            std = torch.zeros_like(z_logvar) + 1e-5
        # std = std / (self.manifold.radius**self.true_dim)  # TODO: Incorporate radius for (P)VMF
        assert torch.isfinite(std).all()
        return z_mean_h, std

    def reparametrize(self, z_mean: Tensor, z_logvar: Tensor) -> Tuple[Q, P]:
        return self.sampling_procedure.reparametrize(z_mean, z_logvar)

    def kl_loss(self, q_z: Q, p_z: P, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return self.sampling_procedure.kl_loss(q_z, p_z, z, data)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(R^{self.dim})"

    def _shortcut(self) -> str:
        return f"{self.__class__.__name__.lower()[0]}{self.true_dim}"

    def summary_name(self, comp_idx: int) -> str:
        return f"comp_{comp_idx:03d}_{self._shortcut()}"

    def summaries(self, comp_idx: int, q_z: Q, prefix: str = "train") -> Dict[str, Tensor]:
        name = prefix + "/" + self.summary_name(comp_idx)
        return {
            name + "/mean/norm": torch.norm(q_z.mean, p=2, dim=-1),
            name + "/stddev/norm": torch.norm(q_z.stddev, p=2, dim=-1),
        }

    def create_manifold(self) -> Manifold:
        return Hyperboloid(lambda: self.radius)


class EuclidianComponent(torch.nn.Module):

    def forward(self, x: Tensor) -> Tuple[Q, P, Tuple[Tensor, ...]]:
        z_params = self.encode(x)
        q_z, p_z = self.reparametrize(*z_params)
        return q_z, p_z, z_params

    def __init__(self, in_dim, out_dim, sampling_procedure: Type[SamplingProcedure[Q, P]]) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._sampling_procedure_type = sampling_procedure
        self.sampling_procedure: SamplingProcedure[Q, P] = None
        self.manifold: Manifold = None

        self.fc_mean = None
        self.fc_logvar = None

    def init_layers(self, scalar_parametrization: bool) -> None:
        self.manifold = self.create_manifold()
        self.sampling_procedure = self._sampling_procedure_type(self.manifold, scalar_parametrization)

        self.fc_mean = torch.nn.Linear(self.in_dim, self.out_dim, dtype=torch.float64)

        if scalar_parametrization:
            self.fc_logvar = torch.nn.Linear(self.in_dim, 1, dtype=torch.float64)
        else:
            self.fc_logvar = torch.nn.Linear(self.in_dim, self.out_dim, dtype=torch.float64)

    @property
    def device(self) -> torch.device:
        return self.fc_logvar.weight.device

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_mean = self.fc_mean(x)
        assert torch.isfinite(z_mean).all()
        z_mean_h = self.manifold.exp_map_mu0(z_mean)
        assert torch.isfinite(z_mean_h).all()

        z_logvar = self.fc_logvar(x)
        assert torch.isfinite(z_logvar).all()
        # +eps prevents collapse
        if self.training:
            std = F.softplus(z_logvar) + 1e-5
        else:
            std = torch.zeros_like(z_logvar) + 1e-5
        # std = std / (self.manifold.radius**self.true_dim)  # TODO: Incorporate radius for (P)VMF
        assert torch.isfinite(std).all()
        return z_mean_h, std

    def reparametrize(self, z_mean: Tensor, z_logvar: Tensor) -> Tuple[Q, P]:
        return self.sampling_procedure.reparametrize(z_mean, z_logvar)

    def kl_loss(self, q_z: Q, p_z: P, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return self.sampling_procedure.kl_loss(q_z, p_z, z, data)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(R^{self.dim})"

    def _shortcut(self) -> str:
        return f"{self.__class__.__name__.lower()[0]}{self.true_dim}"

    def summary_name(self, comp_idx: int) -> str:
        return f"comp_{comp_idx:03d}_{self._shortcut()}"

    def summaries(self, comp_idx: int, q_z: Q, prefix: str = "train") -> Dict[str, Tensor]:
        name = prefix + "/" + self.summary_name(comp_idx)
        return {
            name + "/mean/norm": torch.norm(q_z.mean, p=2, dim=-1),
            name + "/stddev/norm": torch.norm(q_z.stddev, p=2, dim=-1),
        }

    def create_manifold(self) -> Manifold:
        return Euclidean()

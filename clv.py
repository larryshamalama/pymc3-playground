#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import aesara.tensor as at
import numpy as np

from aesara.scalar import Clip
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

from pymc.distributions.distribution import SymbolicDistribution, _get_moment
from pymc.util import check_dist_not_registered


class PoissonProcessRV(RandomVariable):
    name = "poisson_process"
    ndim_supp = 1
    ndim_params = [0]
    dtype = "floatX"
    _print_name = ("ProcessProcess", "\\operatorname{PoissonProcess}")

    def make_node(self, rng, lambda, T, id, T0=0):

        T = at.as_tensor_variable(T)
        T0 = at.as_tensor_variable(T0)

        # T0 and T cannot be random variables
        if T.owner is not None or T0.owner is not None:
            raise ValueError("T and T0 must be scalars, i.e. observed, and not random quantities")

        return super().make_node(rng, T, id, T0)


class PoissonProcess(SymbolicDistribution):
    r"""
    BN-Pareto model for modelling the lifetime of a customer
    """

    @classmethod
    def dist(cls, tx, T, id, T0, **kwargs):
        return super().dist([tx, T, id, T0])

    @classmethod
    def rv_op(cls, tx, T, id, T0, **kwargs):
        rv_out = 
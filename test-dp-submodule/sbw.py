import numpy as np

import aesara.tensor as at
from aesara.tensor.random.op import RandomVariable, default_shape_from_params

import sys

sys.path.insert(0, "/Users/larryshamalama/Documents/GitHub/pymc")

import pymc as pm

from pymc.distributions.continuous import assert_negative_support, UnitContinuous
from pymc.distributions.dist_math import betaln, bound, normal_lcdf
from pymc.distributions.distribution import Continuous

from pymc.distributions import transforms

from pymc.aesaraf import floatX, intX

# a bunch of imports for testing and printing

from aesara.tensor.basic import get_vector_length
from aesara.tensor.random.utils import params_broadcast_shapes
from aesara.tensor.shape import shape_tuple

import aesara

import matplotlib.pyplot as plt
import scipy.stats as st


M = 3; K = 19

class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("StickBreakingWeights", "\\operatorname{StickBreakingWeights}")
    
    def __call__(self, alpha=1., size=None, **kwargs):
        return super().__call__(alpha, size=size, **kwargs)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        return size
    
    @classmethod
    def rng_fn(cls, rng, alpha, size):
        if size is None:
            raise ValueError("size cannot be None")  
        elif isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        
        if np.ndim(alpha) == 0:
            betas = rng.beta(1, alpha, size=size)
        else:
            betas = np.empty(alpha.size + size)

            for idx in np.ndindex(size):
                betas[idx] = rng.beta(1, alpha[idx], size=size)
        
        sticks = np.concatenate(
            (
                np.ones(shape=(size[:-1] + (1,))),
                np.cumprod(1 - betas[..., :-1], axis=-1),
            ),
            axis=-1,
        )
        
        weights = sticks * betas
        weights = np.concatenate(
            (
                weights,
                1 - weights.sum(axis=-1)[..., np.newaxis]
            ),
            axis=-1,
        )

        return weights
    

stickbreakingweights = StickBreakingWeightsRV()


class StickBreakingWeights(Continuous):
    rv_op = stickbreakingweights
    
    def __new__(cls, name, *args, **kwargs):
        kwargs.setdefault("transform", transforms.stick_breaking)
        return super().__new__(cls, name, *args, **kwargs)

    @classmethod
    def dist(cls, alpha, *args, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))

        assert_negative_support(alpha, "alpha", "StickBreakingWeights")

        return super().dist([alpha], **kwargs)

    def logp(value, alpha):
        K = floatX(value.shape[-1])
        
        logp = -at.sum(
            at.log(
                at.cumsum(
                    value[..., ::-1],
                    axis=-1,
                )
            ),
            axis=-1,
        )
        logp -= -K * betaln(1, alpha)
        logp += alpha * at.log(value[..., -1])

        return bound(
            logp,
            alpha > 0,
            at.all(value >= 0),
            at.all(value <= 1),
        )

import numpy as np
import pymc3 as pm # v4.0

from aesara import tensor as at # v2.0.12

import aesara, warnings

from aesara.tensor.random.op import RandomVariable

from pymc3.distributions.continuous import assert_negative_support
from pymc3.distributions.distribution import Continuous


class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0]
    dtype = "floatX"
    _print_name = ("Stick-Breaking W=eights", "\\operatorname{StickBreakingWeights}")

    def __call__(self, alpha, size=None, **kwargs):
        return super().__call__(alpha, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, alpha, size):
        betas = rng.beta(1, alpha, size=size)
        sticks = np.concatenate(
            [
                [1],
                np.cumprod(1 - betas[:-1]),
            ]
        )


        return betas * sticks


stickbreakingweights = StickBreakingWeightsRV()


class StickBreakingWeights(Continuous):
    rv_op = stickbreakingweights

    @classmethod
    def dist(cls, alpha, *args, **kwargs):
        alpha = at.as_tensor_variable(alpha)

        assert_negative_support(alpha, "alpha", "StickBreakingWeights")

        return super().dist([alpha], **kwargs)

    def logp(value, alpha):
        # alpha=1, beta=alpha is confusing... need to revisit this
        return bound(
            at.sum(pm.Beta.logp(value, alpha=1, beta=alpha)),
            alpha > 0,
        )

    def _distr_parameters_for_repr(self):
        return ["alpha"]


if __name__ == "__main__":
    with pm.Model() as model:
        sbw = StickBreakingWeights("test-sticks", alpha=1)
        # larry_norm = LarryNormal(name="larrynormal", mu=1, sigma=2)

        trace = pm.sample(1000)
        print(trace.to_dict()["posterior"]["test-sticks"][0].mean())

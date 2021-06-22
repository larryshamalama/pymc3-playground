import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

from aesara import tensor as at
from matplotlib import pyplot as plt


old_faithful_df = pd.read_csv(pm.get_data("old_faithful.csv"))
waiting_times = old_faithful_df["waiting"]
waiting_times = ((waiting_times - waiting_times.mean())/waiting_times.std()).values

K = 20

def stick_breaking(betas):
    '''
    betas is a K-vector of iid draws from a Beta distribution
    '''
    sticks = at.concatenate(
        [
            [1],
            (1 - betas[:-1]),
        ]
    )
                        
    return at.mul(betas, at.cumprod(sticks))


if __name__ == "__main__":

    with pm.Model() as model:
        alpha = pm.Gamma(name="alpha", alpha=1, beta=1)
        v = pm.Beta(name="v", alpha=1, beta=alpha, shape=(K,)) # beta=alpha kinda confusing here
        
        w = pm.Deterministic(name="w", var=stick_breaking(v))
        mu = pm.Normal(name="mu", mu=0, sigma=5)
        sigma = pm.InverseGamma(name="sigma", alpha=1, beta=1, shape=(K,))
        obs = pm.NormalMixture(name="theta", w=w, mu=mu, tau=1/sigma, observed=waiting_times)

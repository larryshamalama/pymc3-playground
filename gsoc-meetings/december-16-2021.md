## July 29, 2021

#### What I worked on

- Stick-breaking weights PR: https://github.com/pymc-devs/pymc/pull/5200
    - `K` as explicit parameter
    - Need to address Ricardo's suggestions
- Understanding sampling from a DPM posterior in course project

#### Specific questions

- `DirichletProcess` and `DirichletProcessMixture`?
    - Start with some `DPBase`
    - `DirichletProcess` can use `BinaryMetropolisGibbs` and take `M` and `G0` as initialization parameters
    - API design?

```{python}
y = np.array([4, 2, 7, 3])
M = 5

with pm.Model() as model:
    G0 = pm.Poisson("G0", 2)
    dp = pm.DirichletProcess("dp", M, G0, observed=y)
    trace = pm.sample()
```

#### Administrative

- Keep going!
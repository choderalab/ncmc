# One-dimensional toy model to study the effects of neglecting shadow work

To understand what happens when we neglect shadow work, we consider an analytically and numerically tractable one-dimensional toy model system.

The reduced potential has the form
```
u(x;\lambda) = (1/2) (x - \lambda)^2 + \lambda
```
with the parameter `\lambda \in [0,1]` sampling one of two stable states at 0 or 1.

We consider an NCMC simulation in which we perform either un-Metropolized Langevin (VVVR) or Metropolized Langevin (GHMC) dynamics at either endstate, and then attempt trials where `\lambda` is switched from `0 \rightarrow 1` or from `1 \rightarrow 0` using NCMC.
NCMC makes use of either velocity Verlet (VV), un-Metropolized Langevin (VVVR), or Metropolized Langevin (GHMC) dynamics.
During switching, both protocol work and shadow work are accumulated, with the true total work representing the sum.

In the first test, both states `\lambda = 0` and `\lambda = 1` are simulated separately and statistics of the work resulting from switching trials are accumulated.

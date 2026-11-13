# Mathematical documentation

This document states every equation the code solves, how each of the three kinds of solution is
obtained, and how every reported number is defined. Source files are named beside each item.

Contents: [conventions](#conventions) · [the five problems](#the-five-problems) ·
[conventional solvers](#conventional-numerical-solvers) · [PINNs](#physics-informed-neural-networks) ·
[error measures](#error-measures) · [operator learning](#operator-learning) · [references](#references)

## Conventions

Three kinds of solution are kept strictly apart everywhere in the code, the API and the interface:

| Kind | Meaning |
| --- | --- |
| `analytical` | A closed form, a convergent series, or an exact integral representation |
| `numerical` | A conventional discretisation: finite differences or a spectral method |
| `learned` | The output of a neural network trained with PyTorch |

All three are evaluated on the same tensor-product grid, so errors need no interpolation. Fields are
indexed in the order of the coordinates, for example `u[i, j]` is $u(x_i, t_j)$.

## The five problems

Source: `core/neuralphysics/problems/`.

### 1. Heat equation in one dimension (`heat_1d`)

```math
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \qquad x \in (0,1),\; t \in (0,T]
```

Boundary conditions: $u(0,t) = u_L$, $u(1,t) = u_R$. Initial condition: the steady profile plus a shape
that vanishes at both ends, so initial and boundary data are compatible,

```math
u(x,0) = u_L + (u_R - u_L)\,x + s(x), \qquad
s(x) \in \Big\{ A\sin(k\pi x),\;\; A\big[\sin(\pi x) + \tfrac12 \sin(3\pi x)\big],\;\; A\,e^{-(x-1/2)^2/(2\cdot 0.08^2)} \Big\}.
```

**Analytical solution.** Subtracting the steady profile leaves a homogeneous Dirichlet problem, solved by
a Fourier sine series:

```math
u(x,t) = u_L + (u_R - u_L)\,x + \sum_{k=1}^{80} b_k\, e^{-\alpha k^2\pi^2 t}\sin(k\pi x), \qquad
b_k = 2\int_0^1 s(x)\sin(k\pi x)\,\mathrm{d}x .
```

The coefficients are computed with Simpson's rule on 4001 points. For the Gaussian profile the
coefficients decay like $e^{-(k\pi\sigma)^2/2}$, so 80 modes are far more than needed.

### 2. Heat equation in two dimensions (`heat_2d`)

```math
\frac{\partial u}{\partial t} = \alpha\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right),
\qquad (x,y) \in (0,1)^2
```

with $u = u_b$ on the boundary and $u(x,y,0) = u_b + A\sin(m\pi x)\sin(n\pi y)$.

**Analytical solution** (separation of variables):

```math
u(x,y,t) = u_b + A\,e^{-\alpha (m^2+n^2)\pi^2 t}\,\sin(m\pi x)\sin(n\pi y).
```

### 3. Wave equation (`wave_1d`)

```math
\frac{\partial^2 u}{\partial t^2} = c^2\frac{\partial^2 u}{\partial x^2}, \qquad u(0,t) = u(1,t) = 0
```

with $u(x,0) = a_1\sin(k_1\pi x) + a_2\sin(k_2\pi x)$ and $u_t(x,0) = v\,\sin(k_1\pi x)$.

**Analytical solution** (standing waves, $\omega_j = c\,k_j\pi$):

```math
u(x,t) = \sin(k_1\pi x)\Big[a_1\cos(\omega_1 t) + \frac{v}{\omega_1}\sin(\omega_1 t)\Big] + a_2\sin(k_2\pi x)\cos(\omega_2 t).
```

### 4. Viscous Burgers equation (`burgers_1d`)

```math
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2},
\qquad x\in(-1,1),\; t\in(0,1],\qquad u(x,0) = -A\sin(\pi x),\quad u(\pm1,t)=0 .
```

The default $\nu = 0.01/\pi$, $A = 1$ is the standard PINN benchmark (Raissi et al., 2019).

**Analytical solution** (Cole-Hopf transform). With $\Phi(y)=\int_0^y u_0(s)\,\mathrm{d}s = A(\cos\pi y - 1)/\pi$,

```math
u(x,t) = \frac{\displaystyle\int_{-\infty}^{\infty} u_0(y)\,K(x,y,t)\,\mathrm{d}y}{\displaystyle\int_{-\infty}^{\infty} K(x,y,t)\,\mathrm{d}y},
\qquad K = \exp\!\Big(-\frac{(x-y)^2}{4\nu t} - \frac{\Phi(y)}{2\nu}\Big).
```

Both integrands are analytic and decay like a Gaussian, so the trapezoid rule on a truncated window
converges exponentially. The window half-width is $\sqrt{4\nu t\,(\Delta\Phi/2\nu + 40)}$, which leaves
tails below $e^{-40}$ of the peak, and 2001 nodes are used. Exponents are shifted by their maximum before
exponentiation; without that, $e^{\Phi/2\nu}$ overflows at small viscosity. Source:
`solvers/cole_hopf.py`.

**Why a whole-line formula and a periodic solver are valid for a Dirichlet problem.** The initial
condition is odd and 2-periodic. If $u(x,t)$ solves Burgers' equation then so does $-u(-x,t)$, so oddness
is preserved, and so is periodicity. An odd, 2-periodic function satisfies $u(1) = u(-1) = -u(1)$, hence
$u(\pm 1, t) = 0$ for all time.

### 5. Poisson equation (`poisson_2d`)

```math
\nabla^2 u = f \quad\text{in } (0,1)^2, \qquad u = g \quad\text{on the boundary}.
```

**Manufactured solution.**

```math
u(x,y) = A\sin(m\pi x)\sin(n\pi y) + B\,(x^2 - y^2), \qquad f = -A\,(m^2+n^2)\pi^2 \sin(m\pi x)\sin(n\pi y), \qquad g = B\,(x^2-y^2).
```

The second term is harmonic, so changing $B$ changes the boundary data without changing the source.

## Conventional numerical solvers

Source: `core/neuralphysics/solvers/`. Each solver runs on a grid finer than the evaluation grid by an
integer factor and is subsampled, never interpolated.

| Problem | Scheme | Order | Default resolution |
| --- | --- | --- | --- |
| Heat 1D | Crank-Nicolson, tridiagonal solve | 2 in $\Delta x$, 2 in $\Delta t$ | $\Delta x = 1/400$, $\Delta t = T/1000$ |
| Heat 2D | Crank-Nicolson, five-point Laplacian, one sparse LU factorisation | 2, 2 | $h = 1/160$, $\Delta t = T/200$ |
| Wave | Explicit leapfrog with a second-order first step | 2, 2 | $\Delta x = 1/1000$, Courant number $\le 0.5$ |
| Burgers | Fourier pseudo-spectral, ETDRK4, 2/3-rule dealiasing | spectral, 4 | 2048 modes, $\Delta t = 5\cdot10^{-4}$ |
| Poisson | Five-point Laplacian, sparse direct solve | 2 | $h = 1/160$ |

**Crank-Nicolson.** With $r = \alpha\,\Delta t/\Delta x^2$ and $\delta^2$ the second difference,
$(I - \tfrac r2\delta^2)\,u^{n+1} = (I + \tfrac r2\delta^2)\,u^{n}$. It is unconditionally stable.

**Leapfrog.** $u^{n+1}_j = 2u^n_j - u^{n-1}_j + r^2\,\delta^2 u^n_j$ with $r = c\,\Delta t/\Delta x \le 1$. The first
step uses the Taylor expansion $u^1 = u^0 + \Delta t\,v_0 + \tfrac12 r^2\delta^2u^0$. The Courant number is
kept at 0.5 on purpose: at exactly 1 the scheme is exact in one dimension, which would hide its error.

**ETDRK4.** In Fourier space Burgers' equation is $\hat u_t = L\hat u + N(\hat u)$ with
$L = -\nu k^2$ and $N(\hat u) = -\tfrac{\mathrm{i}k}{2}\,\widehat{u^2}$. The linear part is integrated exactly
and the nonlinear part with a fourth-order Runge-Kutta scheme (Cox and Matthews). The coefficient
functions such as $(e^{z}-1)/z$ suffer cancellation for small $z$, so they are evaluated as the mean over
32 points of a unit circle around $z$ (Kassam and Trefethen, 2005).

## Physics-informed neural networks

Source: `core/neuralphysics/pinn/`.

A network $u_\theta$ approximates the solution. Its parameters minimise

```math
\mathcal{L}(\theta) = w_r\,\mathcal{L}_r + w_b\,\mathcal{L}_b + w_0\,\mathcal{L}_0,
```

```math
\mathcal{L}_r = \frac{1}{N_r}\sum_{i=1}^{N_r}\big|\mathcal{N}[u_\theta](\mathbf{x}_i)\big|^2,\quad
\mathcal{L}_b = \frac{1}{N_b}\sum_{i=1}^{N_b}\big|u_\theta(\mathbf{x}^b_i) - g(\mathbf{x}^b_i)\big|^2,\quad
\mathcal{L}_0 = \frac{1}{N_0}\sum_{i=1}^{N_0}\big|u_\theta(\mathbf{x}^0_i) - u_0(\mathbf{x}^0_i)\big|^2 .
```

$\mathcal{N}$ is the differential operator written as a residual, for example
$\mathcal{N}[u] = u_t + u\,u_x - \nu\,u_{xx}$. The wave equation adds
$\frac{1}{N_0}\sum|\partial_t u_\theta - v_0|^2$ with the weight $w_0$. Defaults: $w_r = 1$, $w_b = w_0 = 10$.

**Derivatives.** All derivatives of $u_\theta$ are exact, by reverse-mode automatic differentiation with
respect to the network inputs (`pinn/autodiff.py`). No finite differences are used in training or in the
reported PDE residual of a learned solution.

**Collocation points.** Interior points are a Latin hypercube sample of the domain. Boundary points are
uniform on each face and initial points uniform on $t = 0$. The points are fixed for the whole run.

**Optimisation.** Adam with full-batch gradients (one epoch is one step), with the learning rate
multiplied by 0.9 every 1000 epochs, followed by L-BFGS with a strong-Wolfe line search and a history of
50. Training is in float64 by default because L-BFGS stalls early in float32.

## Error measures

Source: `core/neuralphysics/metrics.py`, `experiments.py`.

For a field $v$ on the $N$ grid points, $\mathrm{rms}(v) = \sqrt{\tfrac1N\sum v_i^2}$ and $\max|v_i|$.

| Reported quantity | Definition | Applies to |
| --- | --- | --- |
| PDE residual | rms and max of $\mathcal{N}[u]$ on the evaluation grid, by automatic differentiation | learned; analytical (as a check that it is near machine zero) |
| Boundary error | rms and max of $u - g$ on all boundary nodes of the grid | all three |
| Initial error | rms and max of $u(\cdot,0) - u_0$ on the $t = 0$ slice | all three |
| Initial velocity error | rms and max of $\partial_t u_\theta(\cdot,0) - v_0$ | learned (wave) |
| Prediction error | relative $L^2$, RMSE and max of $u - u_{\text{ref}}$ | numerical vs analytical; learned vs both |

```math
\varepsilon_{\text{rel}} = \frac{\lVert u - u_{\text{ref}}\rVert_2}{\lVert u_{\text{ref}}\rVert_2}.
```

The PDE residual of a numerical solution is not reported: a discrete scheme satisfies its own difference
equations, and its meaningful accuracy measure is its error against the analytical solution.

## Operator learning

Source: `core/neuralphysics/operators/`. Architectures are described in [ARCHITECTURE.md](ARCHITECTURE.md).

**Task.** Learn the solution operator of the periodic viscous Burgers equation,

```math
G:\; u_0(x) \;\longmapsto\; u(x,t), \qquad x\in[0,1)\ \text{periodic},\quad t\in[0,T].
```

**Inputs.** Zero-mean, band-limited Gaussian random fields with a squared-exponential spectrum,

```math
u_0(x) = \sum_{k=1}^{K} s_k\big(a_k\cos 2\pi kx + b_k\sin 2\pi kx\big),\quad a_k,b_k\sim\mathcal{N}(0,1),\quad
s_k^2 \propto e^{-(2\pi k\ell)^2/2},\quad \textstyle\sum_k s_k^2 = \sigma^2 .
```

Defaults: $K = 12$, length scale $\ell = 0.10$, $\sigma = 0.6$, $\nu = 0.03$, $T = 1$, 64 points in $x$ and
32 time levels. The out-of-distribution set uses $\ell = 0.06$ (rougher inputs). Train, test and
out-of-distribution sets come from three independent random streams of one seed.

**Reference outputs.** The ETDRK4 solver on a grid four times finer in $x$ with $\Delta t \le 10^{-3}$. The
band-limited input is upsampled exactly by zero-padding its spectrum. Halving both step sizes changes
the result by about $10^{-8}$ in relative $L^2$.

**Data loss** (all three models): the mean per-sample relative $L^2$ error over the space-time grid.

**Physics loss** (PINO):

```math
\mathcal{L} = \lambda_d\,\mathcal{L}_{\text{data}} + \lambda_p\,\operatorname{mean}\big(R[u_\theta]^2\big) + \lambda_0\,\operatorname{mean}\big((u_\theta(\cdot,0)-u_0)^2\big),
\qquad R[u] = u_t + u\,u_x - \nu\,u_{xx}.
```

On a grid the derivatives are discrete: $u_x = \mathcal{F}^{-1}(\mathrm{i}k\,\hat u)$ and
$u_{xx} = \mathcal{F}^{-1}(-k^2\hat u)$ exactly for a periodic field, and

```math
u_t(t_n) \approx \frac{-u^{n+2} + 8u^{n+1} - 8u^{n-1} + u^{n-2}}{12\,\Delta t}.
```

**The residual excludes the first time levels, and why.** $R$ is evaluated on levels $3,\dots,n_t-3$. A
Fourier mode $k$ of the initial condition decays at rate $\nu(2\pi k)^2$; for $k \ge 5$ that is faster than
one time step of the grid. A finite difference across the first levels therefore measures an unresolved
transient, not a violation of the PDE: over the first two interior levels the rms residual of the *exact*
data was 0.35, against 0.005 over all later levels. The initial condition is constrained by its own loss term.

**The residual floor.** The residual of the reference data under this discretisation is reported with
every experiment (about $6\cdot10^{-3}$ at the defaults, 1 % of the rms of $u_t$). A predicted field
cannot be shown to satisfy the PDE more accurately than that.

## References

- M. Raissi, P. Perdikaris, G. E. Karniadakis (2019). Physics-informed neural networks. *Journal of Computational Physics* 378, 686-707.
- L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence* 3. [arXiv:1910.03193](https://arxiv.org/abs/1910.03193)
- Z. Li et al. (2021). Fourier Neural Operator for parametric partial differential equations. ICLR. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- Z. Li et al. (2021). Physics-informed neural operator for learning partial differential equations. [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)
- A.-K. Kassam, L. N. Trefethen (2005). Fourth-order time-stepping for stiff PDEs. *SIAM Journal on Scientific Computing* 26. [PDF](https://people.maths.ox.ac.uk/trefethen/fourth-order.pdf)
- S. M. Cox, P. C. Matthews (2002). Exponential time differencing for stiff systems. *Journal of Computational Physics* 176.
- T. Chen, H. Chen (1995). Universal approximation to nonlinear operators by neural networks. *IEEE Transactions on Neural Networks* 6.

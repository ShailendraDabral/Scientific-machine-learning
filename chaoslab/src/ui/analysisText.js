import { tex } from './dom.js';

const d = (s) => tex(s, true);

/** What each measurement means, how it is computed here, and where it stops being reliable. */
export const EXPLAIN = {
  lyapunov: `
<p><b>Largest Lyapunov exponent λ₁.</b> The average exponential rate at which two infinitesimally close trajectories separate:</p>
${d(String.raw`\lvert\delta(t)\rvert \approx \lvert\delta_0\rvert\,e^{\lambda_1 t}`)}
<p>λ₁ &gt; 0 means chaos: errors grow by a factor e every 1/λ₁ time units. λ₁ = 0 is regular (periodic or quasi-periodic) motion; λ₁ &lt; 0 means trajectories converge onto an equilibrium.</p>
<p><b>How it is computed.</b> Benettin's two-trajectory method. A neighbour is kept a distance d₀ from the reference; after every interval τ the separation d<sub>k</sub> is measured and the neighbour is rescaled back to d₀:</p>
${d(String.raw`\lambda_1 = \frac{1}{K\tau}\sum_{k=1}^{K}\ln\frac{d_k}{d_0}`)}
<p>The ± value is the standard error of ten consecutive block averages. It measures statistical scatter only, not bias from the step size.</p>
<p><b>Sensitivity plot.</b> The upper plot shows the raw separation, without rescaling, for three initial offsets. A straight rising line on this log scale is exponential growth; its slope is λ₁/ln 10. Growth stops when the separation reaches the size of the attractor.</p>
<p><b>Limits.</b> The estimate converges slowly, roughly as 1/T; judge it by the flatness of the running value. On a regular orbit the separation grows linearly, so a finite run gives a small positive number that decays towards zero. For the three-body problem the value is a finite-time exponent of that particular orbit.</p>`,

  embedding: `
<p><b>Phase-space reconstruction.</b> Takens' theorem: the geometry of an attractor can be recovered from a single measured quantity by using delayed copies of it as coordinates,</p>
${d(String.raw`\mathbf{X}(t) = \bigl(x(t),\; x(t+\tau),\; x(t+2\tau)\bigr)`)}
<p>The reconstructed object is a smooth, one-to-one image of the true attractor when the embedding dimension m exceeds twice the attractor dimension. It has the same topology and the same Lyapunov exponents, but distorted shape.</p>
<p><b>Choice of delay.</b> τ must be long enough that the coordinates are not redundant and short enough that they are still related. Here τ is the first lag at which the autocorrelation</p>
${d(String.raw`C(k) = \frac{\sum_i (x_i-\bar x)(x_{i+k}-\bar x)}{\sum_i (x_i-\bar x)^2}`)}
<p>falls below 1/e. You can override it.</p>
<p><b>Limits.</b> m = 3 is fixed; it is sufficient for visualising these systems but is not checked by a false-nearest-neighbour test. The autocorrelation criterion is a linear heuristic; mutual information is often preferred for strongly nonlinear signals.</p>`,

  poincare: `
<p><b>Poincaré section.</b> Instead of the whole trajectory, record only the points where it pierces a chosen surface in one direction. This turns a flow in n dimensions into a map in n − 1.</p>
<p><b>Reading it.</b> A finite set of k points is a periodic orbit of period k. A closed curve is quasi-periodic motion on a torus. A scattered set with layered, fractal structure is a chaotic attractor; in a Hamiltonian system, a scattered “sea” between closed curves is chaotic motion.</p>
<p><b>Return map.</b> The lower plot shows one section coordinate against its next value. If the points fall on a single curve, the dynamics are effectively one-dimensional.</p>
<p><b>How crossings are located.</b> A sign change of the section function between two steps brackets a crossing. The crossing is then found by bisection on the cubic Hermite interpolant of that step, which uses the vector field at both ends and is accurate to O(dt⁴). The reported residual is the largest |g| left at the located points. For the driven Duffing oscillator the section is stroboscopic: one point per drive period.</p>
<p><b>Limits.</b> Points inherit the integration error of the trajectory. No section is defined for the three-body problem: its planar phase space is 12-dimensional and a single section is not informative.</p>`,

  compare: `
<p><b>What is compared.</b> Every method integrates the same initial-value problem with the same step.</p>
<p><b>Trajectory difference.</b> The Euclidean distance in state space from a reference solution,</p>
${d(String.raw`e(t) = \lVert \mathbf{y}_{\text{method}}(t) - \mathbf{y}_{\text{ref}}(t) \rVert`)}
<p>The reference is RK4 at a step 16 times smaller. Its own accuracy is measured by repeating it at half that step; the table reports this “reference self-difference”. Errors below it cannot be resolved.</p>
<p><b>Energy error.</b> |E(t) − E₀| / |E₀| for systems whose model conserves energy. A non-symplectic method shows a steady drift; a symplectic method shows a bounded oscillation, because it exactly conserves a nearby “shadow” Hamiltonian.</p>
<p><b>Computational time.</b> Wall-clock time per step, averaged over repeated runs totalling at least 40 ms, measured on this device. It includes the solver's per-step safety checks. Expect variation of several percent between runs.</p>
<p><b>Numerical stability.</b> dt × (fastest local rate of the system) divided by the method's linear stability limit. Above 1 the method amplifies errors on the linear test problem. “Diverged” means the state became non-finite or exceeded 10¹², and the time it happened.</p>
<p><b>Limits.</b> In a chaotic system every error grows as e^{λt}, including that of the reference. Beyond a few Lyapunov times the “difference” saturates at the attractor size and says nothing about method quality. Keep the duration short for chaotic cases.</p>`,

  dtStudy: `
<p><b>Time-step sensitivity.</b> The same problem is solved with dt, dt/2, dt/4, … and the error at the end time is measured against a reference (RK4 at a step 8 times smaller than the smallest dt).</p>
<p><b>Observed order.</b> For a method of order p the global error behaves as</p>
${d(String.raw`e(\Delta t) \approx C\,\Delta t^{\,p} \quad\Longrightarrow\quad \log e = \log C + p \log \Delta t`)}
<p>so the slope on the log–log plot is p. The fitted slope is reported next to the nominal order. Agreement confirms the implementation is correct and that dt is in the asymptotic regime.</p>
<p><b>Work–precision plot.</b> Error against computing time. The curve lowest and furthest left is the most efficient method for a given accuracy.</p>
<p><b>Limits.</b> Points at or below the reference's own resolution, and runs that diverged, are excluded from the fit. If fewer than three points remain, no order is reported. With steps that are too large the error is not yet in the dt^p regime and the slope is meaningless; with very small steps it flattens at round-off.</p>`,
};

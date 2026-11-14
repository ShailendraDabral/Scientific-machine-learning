import { resolveScalarParams } from '../validate.js';

/**
 * Duffing oscillator.
 *   x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)
 * First-order form with state y = [x, v]:
 *   x' = v
 *   v' = -delta v - alpha x - beta x^3 + gamma cos(omega t)
 * The system is non-autonomous; t is supplied by the solver as t0 + n*dt.
 */
const schema = {
  alpha: { default: -1, description: 'linear stiffness' },
  beta: { default: 1, description: 'cubic stiffness' },
  delta: { default: 0.3, min: 0, description: 'linear damping' },
  gamma: { default: 0.5, description: 'drive amplitude' },
  omega: { default: 1.2, min: 0, description: 'drive angular frequency' },
};

function accel(t, y, a, p) {
  const x = y[0];
  a[0] = -p.alpha * x - p.beta * x * x * x + p.gamma * Math.cos(p.omega * t);
}

export const duffing = {
  id: 'duffing',
  name: 'Duffing oscillator',
  kind: 'ode',
  stateLabels: ['x', 'v'],
  paramSchema: schema,
  dimension: () => 2,

  resolveParams(input) {
    const params = resolveScalarParams('duffing', schema, input);
    const warnings = [];
    if (params.beta < 0) {
      warnings.push('duffing: beta < 0 (softening spring): the potential is unbounded below and large-amplitude trajectories escape to infinity.');
    } else if (params.beta === 0 && params.alpha < 0) {
      warnings.push('duffing: beta = 0 with alpha < 0 is an inverted linear oscillator; all generic trajectories are unbounded.');
    }
    return { params, warnings };
  },

  defaultState: () => Float64Array.of(1, 0),

  rhs(t, y, dydt, p) {
    const x = y[0], v = y[1];
    dydt[0] = v;
    dydt[1] = -p.delta * v - p.alpha * x - p.beta * x * x * x + p.gamma * Math.cos(p.omega * t);
  },

  /** Jacobian with respect to (x, v), row-major 2x2. */
  jacobian(t, y, J, p) {
    J[0] = 0; J[1] = 1;
    J[2] = -p.alpha - 3 * p.beta * y[0] * y[0]; J[3] = -p.delta;
  },

  /** Local oscillation frequency plus damping and drive rates. */
  rateScale(t, y, p) {
    return Math.sqrt(Math.abs(p.alpha + 3 * p.beta * y[0] * y[0])) + p.delta + p.omega;
  },

  /**
   * Symplectic methods apply only without damping: with delta > 0 the force
   * depends on velocity and the flow contracts phase-space area.
   */
  separable: {
    isApplicable: (p) => (p.delta === 0
      ? { ok: true }
      : { ok: false, reason: 'delta > 0: the damping force depends on velocity, so the system is not Hamiltonian' }),
    accel,
  },

  /** Potential V(x) = alpha x^2 / 2 + beta x^4 / 4. */
  potential: (x, p) => 0.5 * p.alpha * x * x + 0.25 * p.beta * x * x * x * x,

  /** Mechanical energy is conserved only when undamped and undriven. */
  invariants(p) {
    if (p.delta !== 0 || p.gamma !== 0) return {};
    return { energy: (t, y) => 0.5 * y[1] * y[1] + this.potential(y[0], p) };
  },

  /** Drive period 2 pi / omega (Infinity for omega = 0). */
  drivePeriod: (p) => (p.omega > 0 ? (2 * Math.PI) / p.omega : Infinity),

  assumptions: [
    'A single degree of freedom with unit mass, linear viscous damping (delta >= 0), a stiffness with one cubic correction, and a harmonic drive of fixed amplitude and frequency.',
    'alpha < 0 with beta > 0 is the double-well (buckled beam) case; alpha > 0 with beta > 0 is a hardening spring.',
    'Time is explicit in the drive, so the phase space is (x, v, phase); Poincare sections are taken stroboscopically once per drive period.',
  ],
  limitations: [
    'The cubic term is the first nonlinear term of a symmetric restoring force and is only accurate for moderate amplitudes of the physical system it stands for.',
    'With beta < 0 the potential is unbounded below and trajectories can escape; the solver reports this as a NumericalInstabilityError.',
    'Symplectic integrators are valid only for delta = 0 and are refused otherwise. Energy is a conserved quantity only for delta = 0 and gamma = 0.',
    'For exact stroboscopic sampling choose dt = T / N with integer N; otherwise the section needs interpolation.',
  ],
};

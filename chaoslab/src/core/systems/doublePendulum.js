import { resolveScalarParams } from '../validate.js';

/**
 * Planar double pendulum, full nonlinear equations of motion.
 * State y = [theta1, theta2, omega1, omega2]; angles are measured from the
 * downward vertical, in radians, and are NOT wrapped. With D = theta1 - theta2:
 *
 *   omega1' = [ -g (2 m1 + m2) sin(theta1) - m2 g sin(theta1 - 2 theta2)
 *               - 2 m2 sin(D) (omega2^2 l2 + omega1^2 l1 cos(D)) ]
 *             / [ l1 (2 m1 + m2 - m2 cos(2D)) ]
 *
 *   omega2' = [ 2 sin(D) ( omega1^2 l1 (m1 + m2) + g (m1 + m2) cos(theta1)
 *               + omega2^2 l2 m2 cos(D) ) ]
 *             / [ l2 (2 m1 + m2 - m2 cos(2D)) ]
 *
 * These follow from the Lagrangian
 *   L = (m1+m2) l1^2 w1^2 / 2 + m2 l2^2 w2^2 / 2 + m2 l1 l2 w1 w2 cos(D)
 *       + (m1+m2) g l1 cos(theta1) + m2 g l2 cos(theta2)
 * by solving the two Euler-Lagrange equations for the accelerations.
 * The denominator is >= 2 m1 > 0, so the equations have no singularity.
 */
const schema = {
  m1: { default: 1, min: 0, minExclusive: true, description: 'upper bob mass (kg)' },
  m2: { default: 1, min: 0, minExclusive: true, description: 'lower bob mass (kg)' },
  l1: { default: 1, min: 0, minExclusive: true, description: 'upper rod length (m)' },
  l2: { default: 1, min: 0, minExclusive: true, description: 'lower rod length (m)' },
  g: { default: 9.81, min: 0, description: 'gravitational acceleration (m/s^2)' },
};

export const doublePendulum = {
  id: 'doublePendulum',
  name: 'Double pendulum',
  kind: 'ode',
  stateLabels: ['theta1', 'theta2', 'omega1', 'omega2'],
  paramSchema: schema,
  dimension: () => 4,

  resolveParams(input) {
    return { params: resolveScalarParams('doublePendulum', schema, input), warnings: [] };
  },

  /** Both rods horizontal, at rest: a strongly chaotic release. */
  defaultState: () => Float64Array.of(Math.PI / 2, Math.PI / 2, 0, 0),

  rhs(t, y, dydt, p) {
    const th1 = y[0], th2 = y[1], w1 = y[2], w2 = y[3];
    const { m1, m2, l1, l2, g } = p;
    const d = th1 - th2, sd = Math.sin(d), cd = Math.cos(d);
    const den = 2 * m1 + m2 - m2 * Math.cos(2 * d);
    dydt[0] = w1;
    dydt[1] = w2;
    dydt[2] = (-g * (2 * m1 + m2) * Math.sin(th1) - m2 * g * Math.sin(th1 - 2 * th2)
      - 2 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd)) / (l1 * den);
    dydt[3] = (2 * sd * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * Math.cos(th1)
      + w2 * w2 * l2 * m2 * cd)) / (l2 * den);
  },

  /**
   * Fastest local rate: an upper bound on the small-oscillation frequency
   * (the sum of both squared normal-mode frequencies) plus the rotation rate.
   */
  rateScale(t, y, p) {
    const lin = Math.sqrt((p.g * (p.m1 + p.m2) * (p.l1 + p.l2)) / (p.m1 * p.l1 * p.l2));
    return lin + Math.max(Math.abs(y[2]), Math.abs(y[3]));
  },

  /** Total mechanical energy T + V, conserved exactly by the model. */
  energy(y, p) {
    const { m1, m2, l1, l2, g } = p;
    const th1 = y[0], th2 = y[1], w1 = y[2], w2 = y[3];
    const T = 0.5 * (m1 + m2) * l1 * l1 * w1 * w1 + 0.5 * m2 * l2 * l2 * w2 * w2
      + m2 * l1 * l2 * w1 * w2 * Math.cos(th1 - th2);
    const V = -(m1 + m2) * g * l1 * Math.cos(th1) - m2 * g * l2 * Math.cos(th2);
    return T + V;
  },

  invariants(p) {
    return { energy: (t, y) => this.energy(y, p) };
  },

  // No `separable` entry on purpose: the kinetic energy depends on the angles
  // (through cos(theta1 - theta2)), so H is not of the form T(p) + V(q) and the
  // explicit symplectic Euler / Stormer-Verlet schemes do not apply.

  assumptions: [
    'Two point masses on rigid, massless rods, moving in a vertical plane under uniform gravity.',
    'Frictionless pivots and no air drag: the system is conservative and total energy is constant.',
    'Angles are measured from the downward vertical and are kept unwrapped, so full rotations accumulate multiples of 2 pi.',
    'SI units by default (kg, m, s); masses and lengths must be > 0, g >= 0.',
  ],
  limitations: [
    'Real pendulums have distributed mass, bearing friction and drag; this model overstates how long motion persists and does not include rod inertia.',
    'The Hamiltonian is not separable, so Symplectic Euler and Stormer-Verlet are refused. Forward Euler, midpoint and RK4 do not conserve energy; the relative energy drift reported by the solver is the accuracy measure.',
    'Motion is regular at low energy and chaotic at intermediate and high energy; in chaotic regimes nearby trajectories separate exponentially and long individual trajectories are not reproducible across step sizes or methods.',
    'Fast rotations raise the local rate (|omega|), which tightens the step-size limit.',
  ],
};

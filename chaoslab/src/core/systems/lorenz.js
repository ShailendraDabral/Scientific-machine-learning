import { resolveScalarParams } from '../validate.js';

/**
 * Lorenz system (Lorenz 1963).
 *   dx/dt = sigma (y - x)
 *   dy/dt = x (rho - z) - y
 *   dz/dt = x y - beta z
 */
const schema = {
  sigma: { default: 10, min: 0, minExclusive: true, description: 'Prandtl number' },
  rho: { default: 28, min: 0, description: 'reduced Rayleigh number' },
  beta: { default: 8 / 3, min: 0, minExclusive: true, description: 'geometric factor' },
};

export const lorenz = {
  id: 'lorenz',
  name: 'Lorenz system',
  kind: 'ode',
  stateLabels: ['x', 'y', 'z'],
  paramSchema: schema,
  dimension: () => 3,

  resolveParams(input) {
    const params = resolveScalarParams('lorenz', schema, input);
    const warnings = [];
    if (params.sigma <= params.beta + 1 && params.rho > 1) {
      warnings.push('lorenz: sigma <= beta + 1, so the equilibria C+- are stable for every rho; no chaotic attractor is expected.');
    }
    return { params, warnings };
  },

  defaultState: () => Float64Array.of(1, 1, 1),

  rhs(t, y, dydt, p) {
    const x = y[0], yy = y[1], z = y[2];
    dydt[0] = p.sigma * (yy - x);
    dydt[1] = x * (p.rho - z) - yy;
    dydt[2] = x * yy - p.beta * z;
  },

  /** Analytic Jacobian df/dy, row-major 3x3. */
  jacobian(t, y, J, p) {
    J[0] = -p.sigma; J[1] = p.sigma;   J[2] = 0;
    J[3] = p.rho - y[2]; J[4] = -1;    J[5] = -y[0];
    J[6] = y[1];     J[7] = y[0];      J[8] = -p.beta;
  },

  /** Upper bound on the fastest local rate: infinity norm of the Jacobian. */
  rateScale(t, y, p) {
    const ax = Math.abs(y[0]);
    return Math.max(2 * p.sigma, Math.abs(p.rho - y[2]) + 1 + ax, Math.abs(y[1]) + ax + p.beta);
  },

  invariants: () => ({}),

  /** Equilibria: the origin, and C+- for rho > 1. */
  equilibria(p) {
    const pts = [[0, 0, 0]];
    if (p.rho > 1) {
      const s = Math.sqrt(p.beta * (p.rho - 1));
      pts.push([s, s, p.rho - 1], [-s, -s, p.rho - 1]);
    }
    return pts;
  },

  /** Constant phase-volume contraction rate, the trace of the Jacobian. */
  divergence: (p) => -(p.sigma + 1 + p.beta),

  assumptions: [
    'A three-mode Galerkin truncation of Rayleigh-Benard convection (Boussinesq fluid between two plates); x is convective intensity, y and z are temperature-field modes.',
    'Variables and time are non-dimensional. sigma > 0, beta > 0 and rho >= 0 are required because they are a Prandtl number, a geometric factor and a Rayleigh ratio.',
    'Deterministic and autonomous: no noise and no explicit time dependence.',
  ],
  limitations: [
    'It is not a quantitative model of real convection: the truncation is only justified near onset (rho slightly above 1). Chaos at rho = 28 is a property of the ODE, not of the fluid.',
    'In the chaotic regime (largest Lyapunov exponent about 0.906 at the defaults) round-off grows by e every 1.1 time units. In double precision an individual trajectory is not pointwise meaningful beyond roughly 40 time units; attractor geometry and statistics remain valid.',
    'Explicit methods need dt * (local rate) below their stability limit; Forward Euler diverges for dt around 0.03 and above at the default parameters.',
  ],
};

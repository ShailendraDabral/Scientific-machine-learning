import { resolveScalarParams } from '../validate.js';

/**
 * Rossler system (Rossler 1976).
 *   dx/dt = -y - z
 *   dy/dt = x + a y
 *   dz/dt = b + z (x - c)
 */
const schema = {
  a: { default: 0.2, description: 'linear growth rate in the x-y plane' },
  b: { default: 0.2, description: 'constant injection into z' },
  c: { default: 5.7, description: 'threshold of the z nonlinearity' },
};

export const rossler = {
  id: 'rossler',
  name: 'Rossler system',
  kind: 'ode',
  stateLabels: ['x', 'y', 'z'],
  paramSchema: schema,
  dimension: () => 3,

  resolveParams(input) {
    const params = resolveScalarParams('rossler', schema, input);
    const warnings = [];
    if (params.a === 0) warnings.push('rossler: a = 0 gives no equilibria formula (division by a) and no spiral growth; equilibria() returns none.');
    if (params.c * params.c < 4 * params.a * params.b) {
      warnings.push('rossler: c^2 < 4ab, the system has no equilibria and trajectories generally escape to infinity.');
    }
    if (params.a <= 0 || params.b <= 0 || params.c <= 0) {
      warnings.push('rossler: the studied regime is a, b, c > 0; outside it trajectories may be unbounded.');
    }
    return { params, warnings };
  },

  defaultState: () => Float64Array.of(1, 1, 0),

  rhs(t, y, dydt, p) {
    const x = y[0], yy = y[1], z = y[2];
    dydt[0] = -yy - z;
    dydt[1] = x + p.a * yy;
    dydt[2] = p.b + z * (x - p.c);
  },

  jacobian(t, y, J, p) {
    J[0] = 0;    J[1] = -1;  J[2] = -1;
    J[3] = 1;    J[4] = p.a; J[5] = 0;
    J[6] = y[2]; J[7] = 0;   J[8] = y[0] - p.c;
  },

  rateScale(t, y, p) {
    return Math.max(2, 1 + Math.abs(p.a), Math.abs(y[2]) + Math.abs(y[0] - p.c));
  },

  invariants: () => ({}),

  /** Equilibria: x = (c +- sqrt(c^2 - 4ab))/2, y = -x/a, z = x/a. */
  equilibria(p) {
    const disc = p.c * p.c - 4 * p.a * p.b;
    if (disc < 0 || p.a === 0) return [];
    const s = Math.sqrt(disc);
    return [(p.c - s) / 2, (p.c + s) / 2].map((x) => [x, -x / p.a, x / p.a]);
  },

  assumptions: [
    'A minimal abstract model of chaos with a single quadratic nonlinearity (z*x); it was designed, not derived from a physical system.',
    'Variables and time are non-dimensional; the system is deterministic and autonomous.',
    'The well-studied regime is a, b, c > 0; the defaults a = b = 0.2, c = 5.7 give the classic funnel-free spiral attractor.',
  ],
  limitations: [
    'The attractor has a bounded basin. Initial conditions outside it, and many parameter choices, give trajectories that escape to infinity; the solver stops with a NumericalInstabilityError when that happens.',
    'Phase-volume contraction (a + x - c) depends on position, and the z spikes are fast: the local rate reaches |x - c| + |z|, which sets the step-size limit.',
    'Chaos is weak (largest Lyapunov exponent about 0.071 at the defaults), so long runs are needed before chaotic statistics converge.',
  ],
};

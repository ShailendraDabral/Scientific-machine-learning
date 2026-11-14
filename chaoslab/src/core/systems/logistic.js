import { resolveScalarParams, checkCount } from '../validate.js';
import { InvalidParameterError } from '../errors.js';

/**
 * Logistic map (May 1976).
 *   x_{n+1} = r x_n (1 - x_n)
 * A discrete-time system: it is iterated exactly, no integrator is involved.
 */
const schema = {
  r: { default: 3.9, min: 0, max: 4, description: 'growth rate' },
};

function checkX0(x0) {
  if (typeof x0 !== 'number' || !Number.isFinite(x0) || x0 < 0 || x0 > 1) {
    throw new InvalidParameterError(`logistic: x0 must be a finite number in [0, 1], got ${String(x0)}`);
  }
}

export const logistic = {
  id: 'logistic',
  name: 'Logistic map',
  kind: 'map',
  stateLabels: ['x'],
  paramSchema: schema,
  dimension: () => 1,

  resolveParams(input) {
    return { params: resolveScalarParams('logistic', schema, input), warnings: [] };
  },

  defaultState: () => Float64Array.of(0.2),

  /** One application of the map. */
  map: (x, p) => p.r * x * (1 - x),

  /** Derivative of the map, f'(x) = r (1 - 2x). */
  derivative: (x, p) => p.r * (1 - 2 * x),

  /** Fixed points: 0 and, for r > 1, 1 - 1/r. */
  fixedPoints: (p) => (p.r > 1 ? [0, 1 - 1 / p.r] : [0]),

  /**
   * Orbit x_discard, ..., x_{discard+n-1} as a Float64Array of length n.
   * `discard` iterates are computed first and thrown away (transient).
   */
  orbit({ params, x0 = 0.2, n, discard = 0 }) {
    const { params: p } = this.resolveParams(params);
    checkX0(x0); checkCount('n', n); checkCount('discard', discard);
    const r = p.r;
    let x = x0;
    for (let i = 0; i < discard; i++) x = r * x * (1 - x);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) { out[i] = x; x = r * x * (1 - x); }
    return out;
  },

  /**
   * Lyapunov exponent  lambda = (1/N) sum ln|r (1 - 2 x_n)|  over N iterates
   * after a discarded transient. Returns -Infinity when the orbit lands
   * exactly on the critical point x = 1/2 (superstable cycle).
   */
  lyapunov({ params, x0 = 0.2, n = 100000, discard = 1000 }) {
    const { params: p } = this.resolveParams(params);
    checkX0(x0); checkCount('n', n); checkCount('discard', discard);
    if (n === 0) throw new InvalidParameterError('logistic: n must be > 0 for a Lyapunov estimate');
    const r = p.r;
    let x = x0;
    for (let i = 0; i < discard; i++) x = r * x * (1 - x);
    let sum = 0, c = 0; // Kahan-compensated sum
    for (let i = 0; i < n; i++) {
      const term = Math.log(Math.abs(r * (1 - 2 * x))) - c;
      const s = sum + term;
      c = (s - sum) - term;
      sum = s;
      x = r * x * (1 - x);
    }
    return sum / n;
  },

  assumptions: [
    'Discrete, non-overlapping generations; x is population divided by carrying capacity, so x lies in [0, 1].',
    'r in [0, 4] is required: only then does the map send [0, 1] into itself. Outside this range orbits leave the interval and diverge to minus infinity.',
    'Deterministic, one-dimensional, no age structure, no noise.',
  ],
  limitations: [
    'A conceptual model of period doubling and chaos, not a quantitative population model.',
    'In floating point every orbit is eventually periodic, because only finitely many doubles exist in [0, 1]. Such cycles are typically many millions of iterates long, far beyond the runs used here, but statements about the infinite-time limit are approximations.',
    'In chaotic regimes the error doubles roughly every iterate at r = 4 (lambda = ln 2), so after about 50 iterates the computed x_n is unrelated to the exact orbit of x0; invariant density and Lyapunov exponent remain reliable.',
    'Initial values 0, 1 and exact preimages of fixed points give degenerate orbits that are not typical.',
  ],
};

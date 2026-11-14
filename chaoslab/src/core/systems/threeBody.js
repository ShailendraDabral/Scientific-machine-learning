import { InvalidParameterError, NumericalInstabilityError } from '../errors.js';

/**
 * Newtonian gravitational N-body problem (N = 3 by default).
 *   r_i'' = G sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
 *
 * State layout y = [positions (N*dim), velocities (N*dim)], body by body:
 *   y[i*dim + k]          k-th coordinate of body i
 *   y[N*dim + i*dim + k]  k-th velocity component of body i
 * Parameters: { G, softening (eps), dim (2 or 3), masses: number[] }.
 */
const defaults = { G: 1, softening: 0, dim: 2, masses: [1, 1, 1] };

function resolveParams(input = {}) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new InvalidParameterError('threeBody: params must be an object');
  }
  for (const key of Object.keys(input)) {
    if (!(key in defaults)) throw new InvalidParameterError(`threeBody: unknown parameter "${key}"`);
  }
  const G = input.G ?? defaults.G;
  const softening = input.softening ?? defaults.softening;
  const dim = input.dim ?? defaults.dim;
  const massesIn = input.masses ?? defaults.masses;
  if (typeof G !== 'number' || !Number.isFinite(G) || G <= 0) {
    throw new InvalidParameterError(`threeBody: G must be a finite number > 0, got ${String(G)}`);
  }
  if (typeof softening !== 'number' || !Number.isFinite(softening) || softening < 0) {
    throw new InvalidParameterError(`threeBody: softening must be a finite number >= 0, got ${String(softening)}`);
  }
  if (dim !== 2 && dim !== 3) {
    throw new InvalidParameterError(`threeBody: dim must be 2 or 3, got ${String(dim)}`);
  }
  if (!Array.isArray(massesIn) && !ArrayBuffer.isView(massesIn)) {
    throw new InvalidParameterError('threeBody: masses must be an array');
  }
  if (massesIn.length < 2) {
    throw new InvalidParameterError('threeBody: at least two bodies are required');
  }
  const masses = Float64Array.from(massesIn);
  for (let i = 0; i < masses.length; i++) {
    if (!Number.isFinite(masses[i]) || masses[i] <= 0) {
      throw new InvalidParameterError(`threeBody: mass ${i} must be a finite number > 0, got ${String(massesIn[i])}`);
    }
  }
  const warnings = [];
  if (softening > 0) {
    warnings.push(`threeBody: softening = ${softening} modifies Newtonian gravity at separations of that order; results are not exact point-mass dynamics.`);
  }
  return { params: { G, softening, dim, masses, eps2: softening * softening }, warnings };
}

/** Accelerations of all bodies from positions y[0 .. N*dim). */
function accel(t, y, a, p) {
  const { G, dim, masses, eps2 } = p, N = masses.length;
  for (let i = 0; i < N * dim; i++) a[i] = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      let r2 = eps2;
      for (let k = 0; k < dim; k++) {
        const d = y[j * dim + k] - y[i * dim + k];
        r2 += d * d;
      }
      if (!(r2 > 0)) {
        throw new NumericalInstabilityError(
          `threeBody: bodies ${i} and ${j} coincide (collision singularity) at t = ${t}`, { t, i, j });
      }
      const inv = G / (r2 * Math.sqrt(r2));
      for (let k = 0; k < dim; k++) {
        const d = y[j * dim + k] - y[i * dim + k];
        a[i * dim + k] += masses[j] * inv * d;   // pull of j on i
        a[j * dim + k] -= masses[i] * inv * d;   // equal and opposite (Newton III)
      }
    }
  }
}

const scratch = new Map(); // acceleration buffers for rhs(), keyed by length
function buffer(n) {
  let b = scratch.get(n);
  if (!b) { b = new Float64Array(n); scratch.set(n, b); }
  return b;
}

function energy(y, p) {
  const { G, dim, masses, eps2 } = p, N = masses.length, nq = N * dim;
  let T = 0, V = 0;
  for (let i = 0; i < N; i++) {
    let v2 = 0;
    for (let k = 0; k < dim; k++) v2 += y[nq + i * dim + k] ** 2;
    T += 0.5 * masses[i] * v2;
    for (let j = i + 1; j < N; j++) {
      let r2 = eps2;
      for (let k = 0; k < dim; k++) r2 += (y[j * dim + k] - y[i * dim + k]) ** 2;
      V -= (G * masses[i] * masses[j]) / Math.sqrt(r2);
    }
  }
  return T + V;
}

/** Component k of total linear momentum. */
function momentum(y, p, k) {
  const { dim, masses } = p, N = masses.length, nq = N * dim;
  let s = 0;
  for (let i = 0; i < N; i++) s += masses[i] * y[nq + i * dim + k];
  return s;
}

/** Component of total angular momentum about the origin; axis 2 = z. */
function angularMomentum(y, p, axis) {
  const { dim, masses } = p, N = masses.length, nq = N * dim;
  const a = (axis + 1) % 3, b = (axis + 2) % 3;
  let s = 0;
  for (let i = 0; i < N; i++) {
    const ra = a < dim ? y[i * dim + a] : 0, rb = b < dim ? y[i * dim + b] : 0;
    const va = a < dim ? y[nq + i * dim + a] : 0, vb = b < dim ? y[nq + i * dim + b] : 0;
    s += masses[i] * (ra * vb - rb * va);
  }
  return s;
}

/** Shift positions and velocities in place to the centre-of-mass frame. */
export function toCenterOfMassFrame(y, p) {
  const { dim, masses } = p, N = masses.length, nq = N * dim;
  let M = 0;
  for (let i = 0; i < N; i++) M += masses[i];
  for (let k = 0; k < dim; k++) {
    let cr = 0, cv = 0;
    for (let i = 0; i < N; i++) {
      cr += masses[i] * y[i * dim + k];
      cv += masses[i] * y[nq + i * dim + k];
    }
    cr /= M; cv /= M;
    for (let i = 0; i < N; i++) {
      y[i * dim + k] -= cr;
      y[nq + i * dim + k] -= cv;
    }
  }
  return y;
}

/** Build a state vector from planar positions and velocities [[x, y], ...]. */
function planarState(pos, vel, dim) {
  const N = pos.length, y = new Float64Array(2 * N * dim);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 2; k++) {
      y[i * dim + k] = pos[i][k];
      y[N * dim + i * dim + k] = vel[i][k];
    }
  }
  return y;
}

/** Initial conditions with known behaviour, used for validation and demos. */
export const presets = {
  /** Chenciner-Montgomery figure-eight choreography (initial data: Simo). */
  figureEight({ dim = 2 } = {}) {
    const r1 = [0.97000436, -0.24308753], v3 = [-0.93240737, -0.86473146];
    const pos = [r1, [-r1[0], -r1[1]], [0, 0]];
    const vel = [[-v3[0] / 2, -v3[1] / 2], [-v3[0] / 2, -v3[1] / 2], v3];
    return {
      params: { G: 1, softening: 0, dim, masses: [1, 1, 1] },
      y0: planarState(pos, vel, dim),
      period: 6.32591398,
    };
  },

  /** Lagrange solution: equal masses on a rigidly rotating equilateral triangle. */
  lagrange({ dim = 2, mass = 1, radius = 1, G = 1 } = {}) {
    const side = Math.sqrt(3) * radius;
    const Omega = Math.sqrt((G * 3 * mass) / side ** 3); // Omega^2 = G M / s^3
    const pos = [], vel = [];
    for (let i = 0; i < 3; i++) {
      const phi = (2 * Math.PI * i) / 3 + Math.PI / 2;
      pos.push([radius * Math.cos(phi), radius * Math.sin(phi)]);
      vel.push([-Omega * radius * Math.sin(phi), Omega * radius * Math.cos(phi)]);
    }
    return {
      params: { G, softening: 0, dim, masses: [mass, mass, mass] },
      y0: planarState(pos, vel, dim),
      period: (2 * Math.PI) / Omega,
    };
  },

  /** Burrau's Pythagorean problem: masses 3, 4, 5 at rest on a 3-4-5 triangle. */
  pythagorean({ dim = 2 } = {}) {
    const pos = [[1, 3], [-2, -1], [1, -1]];
    const vel = [[0, 0], [0, 0], [0, 0]];
    return { params: { G: 1, softening: 0, dim, masses: [3, 4, 5] }, y0: planarState(pos, vel, dim) };
  },
};

export const threeBody = {
  id: 'threeBody',
  name: 'Three-body problem',
  kind: 'ode',
  paramSchema: {
    G: { default: 1, min: 0, minExclusive: true, description: 'gravitational constant' },
    softening: { default: 0, min: 0, description: 'Plummer softening length eps' },
    dim: { default: 2, description: 'spatial dimension, 2 or 3' },
    masses: { default: [1, 1, 1], description: 'body masses, each > 0' },
  },
  dimension: (p) => 2 * p.masses.length * p.dim,
  resolveParams,

  /** Figure-eight orbit when the parameters are three equal unit masses. */
  defaultState(p) {
    if (p.masses.length !== 3) {
      throw new InvalidParameterError('threeBody: no default initial state for N != 3; pass y0');
    }
    return presets.figureEight({ dim: p.dim }).y0;
  },

  rhs(t, y, dydt, p) {
    const nq = p.masses.length * p.dim, a = buffer(nq);
    accel(t, y, a, p);
    for (let i = 0; i < nq; i++) {
      dydt[i] = y[nq + i];
      dydt[nq + i] = a[i];
    }
  },

  /** Fastest pairwise orbital frequency, sqrt(G (m_i + m_j) / r^3). */
  rateScale(t, y, p) {
    const { G, dim, masses, eps2 } = p, N = masses.length;
    let w2 = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        let r2 = eps2;
        for (let k = 0; k < dim; k++) r2 += (y[j * dim + k] - y[i * dim + k]) ** 2;
        const w = (G * (masses[i] + masses[j])) / (r2 * Math.sqrt(r2));
        if (w > w2) w2 = w;
      }
    }
    return Math.sqrt(w2);
  },

  /** H = sum p^2/2m + V(q) is separable, so symplectic methods always apply. */
  separable: { isApplicable: () => ({ ok: true }), accel },

  energy,

  invariants(p) {
    const inv = {
      energy: (t, y) => energy(y, p),
      momentumX: (t, y) => momentum(y, p, 0),
      momentumY: (t, y) => momentum(y, p, 1),
      angularMomentumZ: (t, y) => angularMomentum(y, p, 2),
    };
    if (p.dim === 3) {
      inv.momentumZ = (t, y) => momentum(y, p, 2);
      inv.angularMomentumX = (t, y) => angularMomentum(y, p, 0);
      inv.angularMomentumY = (t, y) => angularMomentum(y, p, 1);
    }
    return inv;
  },

  assumptions: [
    'Point masses interacting only through Newtonian inverse-square gravity; no relativity, tides, radiation, collisions or mergers.',
    'An isolated system: total energy, linear momentum and angular momentum are conserved, and the centre of mass moves uniformly.',
    'Units are non-dimensional with G = 1 by default; masses must be > 0. The code accepts any N >= 2 bodies in 2 or 3 dimensions.',
    'Softening eps = 0 by default (exact Newtonian force). A non-zero eps replaces r^2 by r^2 + eps^2 and is reported as a warning because it changes the physics.',
  ],
  limitations: [
    'The force is singular at zero separation. Close encounters need steps far smaller than the rest of the orbit; with a fixed step the solver warns when dt * sqrt(G (m_i + m_j) / r^3) exceeds the stability limit, and an exact collision raises a NumericalInstabilityError. No regularisation is implemented.',
    'Generic three-body motion is chaotic and often ends with one body escaping; long individual trajectories are not reproducible across step sizes or methods.',
    'Stormer-Verlet and Symplectic Euler keep the energy error bounded and conserve momentum and angular momentum to round-off, but only at fixed step. Euler, midpoint and RK4 show secular energy drift.',
    'Positions grow without bound after an escape, which reduces floating-point resolution of the remaining pair.',
  ],
};

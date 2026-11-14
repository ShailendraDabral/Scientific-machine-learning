import { InvalidParameterError, NumericalInstabilityError } from './errors.js';
import { checkState, checkPositive, checkCount } from './validate.js';
import { getIntegrator } from './integrators/index.js';

/**
 * Fixed-step ODE driver. Pure computation: no DOM, no timers, no rendering.
 *
 * createStepper(system, options) -> stepper
 *   options: {
 *     params,            system parameters (validated; defaults filled in)
 *     y0,                initial state (default: system.defaultState)
 *     t0 = 0,            initial time
 *     dt,                step size, > 0 (required)
 *     method = 'rk4',    'euler' | 'midpoint' | 'rk4' | 'symplecticEuler' | 'verlet'
 *     maxNorm = 1e12,    blow-up threshold on max |y_i|
 *     checkEvery = 16,   steps between stability / invariant checks
 *     strict = false,    throw (instead of warn) when dt exceeds the stability limit
 *   }
 *
 * Guarantees:
 *  - parameters, state, dt are validated before the first step;
 *  - every step is checked for NaN / Infinity / blow-up and throws
 *    NumericalInstabilityError rather than returning garbage;
 *  - dt * (local rate of the system) is compared with the method's linear
 *    stability limit and reported;
 *  - time is computed as t0 + n*dt (never accumulated), so it carries no
 *    summation error;
 *  - conserved quantities, where the model has them, are tracked and their
 *    drift reported.
 */
export function createStepper(system, options = {}) {
  if (!system || system.kind !== 'ode') {
    throw new InvalidParameterError('createStepper: system must be an ODE system (kind "ode")');
  }
  const { params, warnings } = system.resolveParams(options.params);
  const method = getIntegrator(options.method ?? 'rk4');
  const dt = options.dt;
  checkPositive('dt', dt);
  const t0 = options.t0 ?? 0;
  if (typeof t0 !== 'number' || !Number.isFinite(t0)) {
    throw new InvalidParameterError(`t0 must be a finite number, got ${String(t0)}`);
  }
  const maxNorm = options.maxNorm ?? 1e12;
  checkPositive('maxNorm', maxNorm);
  const checkEvery = options.checkEvery ?? 16;
  if (!Number.isInteger(checkEvery) || checkEvery < 1) {
    throw new InvalidParameterError('checkEvery must be an integer >= 1');
  }
  const strict = options.strict ?? false;

  const n = system.dimension(params);
  const y0 = options.y0 ?? system.defaultState(params);
  checkState(system.id, y0, n);

  if (method.requiresSeparable) {
    if (!system.separable) {
      throw new InvalidParameterError(
        `${method.name} needs a separable Hamiltonian H = T(p) + V(q); ${system.name} is not of that form`);
    }
    const ok = system.separable.isApplicable(params);
    if (!ok.ok) {
      throw new InvalidParameterError(`${method.name} cannot be used for ${system.name}: ${ok.reason}`);
    }
  }

  let y = Float64Array.from(y0);
  let out = new Float64Array(n);
  const work = method.createWork(n);
  let steps = 0, t = t0;
  let maxDtRate = 0, rateWarned = false;

  const invFns = system.invariants ? system.invariants(params) : {};
  const inv = {};
  for (const [name, fn] of Object.entries(invFns)) {
    const v = fn(t, y);
    inv[name] = { fn, initial: v, current: v, maxAbsDrift: 0 };
  }

  function checkRate() {
    const z = dt * system.rateScale(t, y, params);
    if (z > maxDtRate) maxDtRate = z;
    if (z > method.stabilityLimit) {
      const msg = `${system.id}/${method.id}: dt * rate = ${z.toPrecision(3)} exceeds the stability limit `
        + `${method.stabilityLimit} at t = ${t.toPrecision(6)}; reduce dt below `
        + `${(dt * method.stabilityLimit / z).toPrecision(3)}`;
      if (strict) throw new NumericalInstabilityError(msg, { t, step: steps, dtRate: z });
      if (!rateWarned) { warnings.push(msg); rateWarned = true; }
    }
  }

  function trackInvariants() {
    for (const rec of Object.values(inv)) {
      rec.current = rec.fn(t, y);
      const d = Math.abs(rec.current - rec.initial);
      if (d > rec.maxAbsDrift) rec.maxAbsDrift = d;
    }
  }

  checkRate();

  return {
    system, method, params, dt, t0, warnings,
    /** Current time, t0 + steps*dt. */
    get t() { return t; },
    /** Current state. The buffer is reused: copy it if you need to keep it. */
    get y() { return y; },
    get steps() { return steps; },

    /** Advance by `count` steps. */
    advance(count = 1) {
      checkCount('count', count);
      for (let s = 0; s < count; s++) {
        method.step(system, params, t, y, dt, out, work);
        const tmp = y; y = out; out = tmp;
        steps++;
        t = t0 + steps * dt;
        for (let i = 0; i < n; i++) {
          // One comparison rejects NaN, +-Infinity and runaway magnitudes.
          if (!(Math.abs(y[i]) <= maxNorm)) {
            throw new NumericalInstabilityError(
              `${system.id}/${method.id}: state component ${i} became ${y[i]} at step ${steps} `
              + `(t = ${t.toPrecision(6)}, dt = ${dt}). The step is too large for this method, `
              + 'or the trajectory is genuinely unbounded.',
              { t, step: steps, index: i, value: y[i] });
          }
        }
        if (steps % checkEvery === 0) { checkRate(); trackInvariants(); }
      }
      checkRate(); trackInvariants();
      return this;
    },

    /** Stability and conservation diagnostics for the run so far. */
    report() {
      const invariants = {};
      for (const [name, rec] of Object.entries(inv)) {
        invariants[name] = {
          initial: rec.initial,
          current: rec.current,
          maxAbsDrift: rec.maxAbsDrift,
          // Relative drift is undefined for quantities that start at zero.
          maxRelDrift: Math.abs(rec.initial) > 1e-12 ? rec.maxAbsDrift / Math.abs(rec.initial) : null,
        };
      }
      return {
        steps, t,
        stability: {
          maxDtRate,
          limit: method.stabilityLimit,
          ok: maxDtRate <= method.stabilityLimit,
        },
        invariants,
        warnings: warnings.slice(),
      };
    },
  };
}

/**
 * Integrate for a fixed number of steps and return the sampled trajectory.
 * options: as createStepper, plus { steps, sampleEvery = 1 }.
 * Returns { t: Float64Array(rows), y: Float64Array(rows*n) row-major, n, rows,
 *           tFinal, final: Float64Array(n), steps, stability, invariants, warnings }.
 */
export function integrate(system, options = {}) {
  const { steps, sampleEvery = 1 } = options;
  checkCount('steps', steps);
  if (!Number.isInteger(sampleEvery) || sampleEvery < 1) {
    throw new InvalidParameterError('sampleEvery must be an integer >= 1');
  }
  const st = createStepper(system, options);
  const n = st.y.length;
  const rows = Math.floor(steps / sampleEvery) + 1;
  const ts = new Float64Array(rows), ys = new Float64Array(rows * n);
  ts[0] = st.t; ys.set(st.y, 0);
  for (let r = 1; r < rows; r++) {
    st.advance(sampleEvery);
    ts[r] = st.t; ys.set(st.y, r * n);
  }
  st.advance(steps - (rows - 1) * sampleEvery);
  const report = st.report();
  return { ...report, t: ts, y: ys, n, rows, tFinal: report.t, final: Float64Array.from(st.y) };
}

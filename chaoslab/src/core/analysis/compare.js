import { createStepper } from '../solve.js';
import { InvalidParameterError } from '../errors.js';
import { integrators } from '../integrators/index.js';

const now = () => performance.now();

function energyFn(system, params) {
  const inv = system.invariants ? system.invariants(system.resolveParams(params).params) : {};
  return inv.energy || null;
}

/** Integrate and sample every `stride` steps; never throws on instability. */
function sampledRun(system, { params, y0, method, dt }, stride, rows, energy) {
  const st = createStepper(system, { params, y0, method, dt });
  const n = st.y.length;
  const y = new Float64Array(rows * n), E = energy ? new Float64Array(rows) : null;
  const E0 = energy ? energy(st.t, st.y) : null;
  let done = 0, failure = null;
  try {
    for (; done < rows; done++) {
      st.advance(stride);
      y.set(st.y, done * n);
      if (E) E[done] = energy(st.t, st.y);
    }
  } catch (e) {
    failure = { name: e.name, message: e.message, t: st.t };
  }
  return { y, E, E0, n, rows: done, failure, report: st.report() };
}

/** Wall-clock cost per step, averaged over repeated runs of at least 40 ms in total. */
function timePerStep(system, opts, steps) {
  try {
    createStepper(system, opts).advance(Math.min(steps, 300)); // warm-up so the JIT has compiled the loop
    let elapsed = 0, reps = 0;
    do {
      const st = createStepper(system, opts);
      const s = now();
      st.advance(steps);
      elapsed += now() - s;
      reps++;
    } while (elapsed < 40 && reps < 400);
    return (elapsed * 1e6) / (reps * steps); // nanoseconds
  } catch {
    return null; // the run does not complete: no meaningful timing
  }
}

function applicability(system, params, method) {
  try {
    createStepper(system, { params, method, dt: 1e-3 });
    return { ok: true };
  } catch (e) {
    if (e instanceof InvalidParameterError) return { ok: false, reason: e.message };
    throw e;
  }
}

function rowDistance(a, b, row, n) {
  let s = 0;
  for (let i = row * n; i < (row + 1) * n; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

/**
 * Compare integrators on one initial-value problem at the same step size.
 *
 * The reference solution is RK4 at dt / refFactor. Its own accuracy is
 * estimated by repeating it at half that step; errors smaller than this
 * "reference self-difference" cannot be resolved and are reported as such.
 */
export function compareMethods(system, { params, y0, dt, duration, methods, refFactor = 16, samples = 300 }) {
  if (!(duration > 0)) throw new InvalidParameterError('duration must be > 0');
  const steps = Math.round(duration / dt);
  if (steps < 4) throw new InvalidParameterError('duration is shorter than four steps');
  const stride = Math.max(1, Math.ceil(steps / samples)), rows = Math.floor(steps / stride);
  const energy = energyFn(system, params);
  const base = { params, y0 };
  const ref = sampledRun(system, { ...base, method: 'rk4', dt: dt / refFactor }, stride * refFactor, rows, null);
  if (ref.failure) throw new Error(`reference solution failed: ${ref.failure.message}`);
  const ref2 = sampledRun(system, { ...base, method: 'rk4', dt: dt / (2 * refFactor) }, stride * refFactor * 2, rows, null);
  const n = ref.n;
  let refSelf = 0;
  if (!ref2.failure) for (let r = 0; r < rows; r++) refSelf = Math.max(refSelf, rowDistance(ref.y, ref2.y, r, n));
  const t = Array.from({ length: rows }, (_, r) => (r + 1) * stride * dt);
  const eScale = energy ? (Math.abs(sampledRun(system, { ...base, method: 'rk4', dt }, 1, 0, energy).E0) > 1e-12) : false;

  const results = methods.map((method) => {
    const meta = integrators[method];
    const app = applicability(system, params, method);
    if (!app.ok) return { method, applicable: false, reason: app.reason };
    const run = sampledRun(system, { ...base, method, dt }, stride, rows, energy);
    const err = [], eErr = [];
    let maxErr = 0, maxE = 0;
    for (let r = 0; r < run.rows; r++) {
      const e = rowDistance(run.y, ref.y, r, n);
      err.push(e); maxErr = Math.max(maxErr, e);
      if (energy) {
        const de = Math.abs(run.E[r] - run.E0) / (eScale ? Math.abs(run.E0) : 1);
        eErr.push(de); maxE = Math.max(maxE, de);
      }
    }
    const ns = run.failure ? null : timePerStep(system, { ...base, method, dt }, steps);
    return {
      method, applicable: true, order: meta.order, symplectic: meta.symplectic, evaluations: meta.evaluationsPerStep,
      status: run.failure ? 'diverged' : (run.report.stability.ok ? 'stable' : 'completed above the linear limit'),
      failure: run.failure, completedRows: run.rows,
      maxDtRate: run.report.stability.maxDtRate, limit: run.report.stability.limit,
      err, maxErr, finalErr: run.rows ? err[run.rows - 1] : null,
      energyErr: energy ? eErr : null, maxEnergyErr: energy ? maxE : null,
      nsPerStep: ns, runMs: ns === null ? null : (ns * steps) / 1e6, steps,
    };
  });
  return {
    t, results, dt, duration: steps * dt, steps,
    hasEnergy: !!energy, energyRelative: eScale,
    reference: { method: 'rk4', dt: dt / refFactor, selfDifference: ref2.failure ? null : refSelf },
  };
}

/**
 * Time-step sensitivity: the same problem solved with dt, dt/2, dt/4, ...
 * The error at the end time is measured against RK4 at a step `refFactor`
 * times smaller than the smallest dt. For a method of order p the error
 * scales as dt^p, so the slope of log(error) against log(dt) is the
 * observed order. Only points above the resolution of the reference and
 * from runs that completed are used in the fit.
 */
export function dtStudy(system, { params, y0, dt, duration, methods, levels = 6, refFactor = 8 }) {
  const steps0 = Math.round(duration / dt);
  if (steps0 < 4) throw new InvalidParameterError('duration is shorter than four steps');
  const T = steps0 * dt;
  const stepsAt = Array.from({ length: levels }, (_, k) => steps0 * 2 ** k);
  const dts = stepsAt.map((s) => T / s);
  const energy = energyFn(system, params);
  const base = { params, y0 };
  const final = (method, steps) => {
    const st = createStepper(system, { ...base, method, dt: T / steps });
    const E0 = energy ? energy(st.t, st.y) : null;
    let maxE = 0;
    const chunk = Math.max(1, Math.floor(steps / 64));
    for (let done = 0; done < steps; done += chunk) {
      st.advance(Math.min(chunk, steps - done));
      if (energy) maxE = Math.max(maxE, Math.abs(energy(st.t, st.y) - E0) / (Math.abs(E0) > 1e-12 ? Math.abs(E0) : 1));
    }
    return { y: Float64Array.from(st.y), maxE };
  };
  const refSteps = stepsAt[levels - 1] * refFactor;
  const ref = final('rk4', refSteps).y;
  const ref2 = final('rk4', refSteps * 2).y;
  const dist = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
  const refSelf = dist(ref, ref2);
  const floor = Math.max(10 * refSelf, 1e-13);

  const results = methods.map((method) => {
    const app = applicability(system, params, method);
    if (!app.ok) return { method, applicable: false, reason: app.reason };
    const errors = [], energies = [], failures = [];
    for (const s of stepsAt) {
      try {
        const r = final(method, s);
        errors.push(dist(r.y, ref)); energies.push(energy ? r.maxE : null); failures.push(null);
      } catch (e) {
        errors.push(null); energies.push(null); failures.push(e.name);
      }
    }
    // least-squares slope of log(error) vs log(dt) over resolvable points
    const xs = [], ys = [];
    errors.forEach((e, i) => { if (e !== null && e > floor) { xs.push(Math.log(dts[i])); ys.push(Math.log(e)); } });
    let order = null;
    if (xs.length >= 3) {
      const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
      let sxy = 0, sxx = 0;
      xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
      order = sxy / sxx;
    }
    const ns = failures[levels - 1] ? null : timePerStep(system, { ...base, method, dt: dts[levels - 1] }, Math.min(stepsAt[levels - 1], 20000));
    return {
      method, applicable: true, nominalOrder: integrators[method].order, symplectic: integrators[method].symplectic,
      errors, energies, failures, order, fitPoints: xs.length, nsPerStep: ns,
      costMs: stepsAt.map((s) => (ns === null ? null : (ns * s) / 1e6)),
    };
  });
  return { dts, stepsAt, duration: T, results, hasEnergy: !!energy, floor,
    reference: { method: 'rk4', dt: T / refSteps, selfDifference: refSelf } };
}

import { createStepper } from '../solve.js';
import { InvalidParameterError } from '../errors.js';

function distance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (b[i] - a[i]) ** 2;
  return Math.sqrt(s);
}

/** Run the transient, then return two steppers that share the same clock. */
function afterTransient(system, { params, y0, method, dt, transient = 0 }) {
  const pre = createStepper(system, { params, y0, method, dt });
  const nT = Math.round(transient / dt);
  pre.advance(nT);
  return { start: Float64Array.from(pre.y), t0: pre.t };
}

/**
 * Largest Lyapunov exponent by the two-trajectory method of Benettin et al.
 *
 * A reference trajectory and a neighbour a distance d0 away are integrated
 * together. Every `renormTime` the separation d is measured, ln(d/d0) is
 * accumulated, and the neighbour is pulled back to distance d0 along the
 * current separation direction. Then
 *     lambda = (1 / T) * sum_k ln(d_k / d0).
 *
 * Returns the estimate, its running value (to judge convergence) and a
 * standard error from the scatter of 10 consecutive block averages.
 */
export function largestLyapunov(system, opts) {
  const { params, method = 'rk4', dt, duration, renormTime, d0 = 1e-8, perturbIndex = 0 } = opts;
  if (!(duration > 0)) throw new InvalidParameterError('duration must be > 0');
  if (!(d0 > 0)) throw new InvalidParameterError('d0 must be > 0');
  const { start, t0 } = afterTransient(system, opts);
  const k = Math.max(1, Math.round((renormTime ?? 10 * dt) / dt));
  const intervals = Math.floor(duration / (k * dt));
  if (intervals < 20) throw new InvalidParameterError('duration is too short: fewer than 20 renormalisation intervals');

  const a = createStepper(system, { params, y0: start, method, dt, t0 });
  const shifted = Float64Array.from(start);
  shifted[perturbIndex] += d0;
  const b = createStepper(system, { params, y0: shifted, method, dt, t0 });

  const logs = new Float64Array(intervals);
  const stride = Math.max(1, Math.ceil(intervals / 400));
  const series = { t: [], lambda: [] };
  let sum = 0;
  for (let i = 0; i < intervals; i++) {
    a.advance(k); b.advance(k);
    const ya = a.y, yb = b.y;
    const d = distance(ya, yb);
    if (!(d > 0)) { // trajectories coincided to round-off: re-seed, count no growth
      yb.set(ya); yb[perturbIndex] += d0; logs[i] = 0;
    } else {
      logs[i] = Math.log(d / d0);
      const s = d0 / d;
      for (let j = 0; j < ya.length; j++) yb[j] = ya[j] + s * (yb[j] - ya[j]);
    }
    sum += logs[i];
    if ((i + 1) % stride === 0 || i === intervals - 1) {
      const elapsed = (i + 1) * k * dt;
      series.t.push(elapsed); series.lambda.push(sum / elapsed);
    }
  }
  const T = intervals * k * dt;
  const lambda = sum / T;

  // Standard error from 10 block means of the local growth rate.
  const blocks = 10, per = Math.floor(intervals / blocks), means = [];
  for (let bI = 0; bI < blocks; bI++) {
    let s = 0;
    for (let i = bI * per; i < (bI + 1) * per; i++) s += logs[i];
    means.push(s / (per * k * dt));
  }
  const m = means.reduce((x, y) => x + y, 0) / blocks;
  const variance = means.reduce((x, y) => x + (y - m) ** 2, 0) / (blocks - 1);
  return {
    lambda, stderr: Math.sqrt(variance / blocks), series, intervals,
    renormTime: k * dt, duration: T, d0, transient: t0, method, dt,
    stability: a.report().stability,
  };
}

/**
 * Sensitivity to initial conditions: separation between a reference
 * trajectory and neighbours started `delta` away, with no renormalisation.
 * Returns log10 of the separation against time for each delta.
 */
export function divergence(system, opts) {
  const { params, method = 'rk4', dt, duration, deltas = [1e-9, 1e-6, 1e-3], perturbIndex = 0, maxSamples = 500 } = opts;
  if (!(duration > 0)) throw new InvalidParameterError('duration must be > 0');
  const { start, t0 } = afterTransient(system, opts);
  const a = createStepper(system, { params, y0: start, method, dt, t0 });
  const bs = deltas.map((delta) => {
    const y = Float64Array.from(start);
    y[perturbIndex] += delta;
    return createStepper(system, { params, y0: y, method, dt, t0 });
  });
  const steps = Math.round(duration / dt), stride = Math.max(1, Math.ceil(steps / maxSamples));
  const t = [0], series = deltas.map((delta) => ({ delta, log10d: [Math.log10(delta)] }));
  for (let done = 0; done < steps; done += stride) {
    a.advance(stride);
    t.push(a.t - t0);
    bs.forEach((b, i) => {
      b.advance(stride);
      const d = distance(a.y, b.y);
      series[i].log10d.push(d > 0 ? Math.log10(d) : null);
    });
  }
  return { t, series, transient: t0, method, dt };
}

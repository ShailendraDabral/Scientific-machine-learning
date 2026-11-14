import { createStepper } from '../solve.js';
import { InvalidParameterError } from '../errors.js';

/** Uniformly sampled scalar observable y[index](t) after a discarded transient. */
export function sampleObservable(system, { params, y0, method = 'rk4', dt, transient = 0, duration, sampleDt, index = 0 }) {
  const st = createStepper(system, { params, y0, method, dt });
  st.advance(Math.round(transient / dt));
  const stride = Math.max(1, Math.round(sampleDt / dt));
  const n = Math.floor(duration / (stride * dt));
  if (n < 50) throw new InvalidParameterError('duration is too short: fewer than 50 samples');
  const x = new Float64Array(n), full = new Float64Array(n * st.y.length);
  for (let i = 0; i < n; i++) {
    st.advance(stride);
    x[i] = st.y[index];
    full.set(st.y, i * st.y.length);
  }
  return { x, full, n: st.y.length, sampleDt: stride * dt };
}

/** Normalised autocorrelation C(k) = <(x_i - m)(x_{i+k} - m)> / <(x_i - m)^2>, k = 0..maxLag. */
export function autocorrelation(x, maxLag) {
  const N = x.length;
  let mean = 0;
  for (let i = 0; i < N; i++) mean += x[i];
  mean /= N;
  let c0 = 0;
  for (let i = 0; i < N; i++) c0 += (x[i] - mean) ** 2;
  const out = new Float64Array(maxLag + 1);
  for (let k = 0; k <= maxLag; k++) {
    let s = 0;
    for (let i = 0; i + k < N; i++) s += (x[i] - mean) * (x[i + k] - mean);
    out[k] = c0 > 0 ? s / c0 : 0;
  }
  return out;
}

/**
 * Delay for embedding: the first lag at which the autocorrelation falls
 * below 1/e. If it never does within the window, the first local minimum,
 * and failing that a quarter of the window.
 */
export function chooseDelay(acf) {
  for (let k = 1; k < acf.length; k++) if (acf[k] < 1 / Math.E) return { lag: k, rule: 'first lag with C < 1/e' };
  for (let k = 1; k < acf.length - 1; k++) {
    if (acf[k] < acf[k - 1] && acf[k] <= acf[k + 1]) return { lag: k, rule: 'first local minimum of C' };
  }
  return { lag: Math.max(1, acf.length >> 2), rule: 'fallback: quarter of the lag window' };
}

/** Delay vectors [x_i, x_{i+lag}, ..., x_{i+(dim-1)lag}] as `dim` columns. */
export function delayEmbed(x, lag, dim = 3) {
  if (!Number.isInteger(lag) || lag < 1) throw new InvalidParameterError('lag must be an integer >= 1');
  const M = x.length - (dim - 1) * lag;
  if (M < 2) throw new InvalidParameterError('series too short for this delay');
  return Array.from({ length: dim }, (_, d) => Array.from(x.subarray(d * lag, d * lag + M)));
}

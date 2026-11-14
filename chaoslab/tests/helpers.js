/** Shared test fixtures. */

/** Harmonic oscillator x'' = -w^2 x with exact solution; a full system object. */
export const oscillator = {
  id: 'oscillator', name: 'Harmonic oscillator', kind: 'ode',
  dimension: () => 2,
  resolveParams: (p = {}) => ({ params: { w: p.w ?? 1 }, warnings: [] }),
  defaultState: () => Float64Array.of(1, 0),
  rhs(t, y, d, p) { d[0] = y[1]; d[1] = -p.w * p.w * y[0]; },
  rateScale: (t, y, p) => p.w,
  separable: { isApplicable: () => ({ ok: true }), accel(t, y, a, p) { a[0] = -p.w * p.w * y[0]; } },
  invariants: (p) => ({ energy: (t, y) => 0.5 * y[1] * y[1] + 0.5 * p.w * p.w * y[0] * y[0] }),
};

/** Central finite-difference Jacobian of system.rhs at (t, y). */
export function fdJacobian(system, t, y, p, h = 1e-6) {
  const n = y.length, J = new Float64Array(n * n);
  const yp = Float64Array.from(y), fp = new Float64Array(n), fm = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    yp[j] = y[j] + h; system.rhs(t, yp, fp, p);
    yp[j] = y[j] - h; system.rhs(t, yp, fm, p);
    yp[j] = y[j];
    for (let i = 0; i < n; i++) J[i * n + j] = (fp[i] - fm[i]) / (2 * h);
  }
  return J;
}

export function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

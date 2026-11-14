/**
 * Forward (explicit) Euler.
 *   y_{n+1} = y_n + h f(t_n, y_n)
 * Order 1. Local error O(h^2), global error O(h).
 * Linear stability on y' = lambda*y requires |1 + h*lambda| <= 1, i.e.
 * h*|lambda| <= 2 on the negative real axis; it is unstable for every h on
 * the imaginary axis (undamped oscillations gain energy each step).
 */
export const euler = {
  id: 'euler',
  name: 'Forward Euler',
  order: 1,
  symplectic: false,
  requiresSeparable: false,
  evaluationsPerStep: 1,
  stabilityLimit: 2.0,
  notes: 'Explicit, first order. Gains energy on oscillatory problems at any step size; provided as a baseline.',
  createWork: (n) => ({ k1: new Float64Array(n) }),
  step(sys, p, t, y, h, out, w) {
    const n = y.length, k1 = w.k1;
    sys.rhs(t, y, k1, p);
    for (let i = 0; i < n; i++) out[i] = y[i] + h * k1[i];
  },
};

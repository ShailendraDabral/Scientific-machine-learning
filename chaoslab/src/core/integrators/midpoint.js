/**
 * Explicit midpoint method (second-order Runge-Kutta).
 *   k1 = f(t_n, y_n)
 *   k2 = f(t_n + h/2, y_n + (h/2) k1)
 *   y_{n+1} = y_n + h k2
 * Order 2. Local error O(h^3), global error O(h^2).
 * Stability function R(z) = 1 + z + z^2/2: real-axis limit h*|lambda| <= 2;
 * on the imaginary axis |R(iy)|^2 = 1 + y^4/4 > 1, so undamped oscillations
 * still grow, though far more slowly than with Euler.
 */
export const midpoint = {
  id: 'midpoint',
  name: 'Explicit midpoint (RK2)',
  order: 2,
  symplectic: false,
  requiresSeparable: false,
  evaluationsPerStep: 2,
  stabilityLimit: 2.0,
  notes: 'Explicit, second order. Weakly unstable for purely oscillatory problems (amplitude grows as h^4 per step).',
  createWork: (n) => ({ k1: new Float64Array(n), k2: new Float64Array(n), tmp: new Float64Array(n) }),
  step(sys, p, t, y, h, out, w) {
    const n = y.length, { k1, k2, tmp } = w, hh = 0.5 * h;
    sys.rhs(t, y, k1, p);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hh * k1[i];
    sys.rhs(t + hh, tmp, k2, p);
    for (let i = 0; i < n; i++) out[i] = y[i] + h * k2[i];
  },
};

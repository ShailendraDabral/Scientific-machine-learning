/**
 * Classical fourth-order Runge-Kutta.
 *   k1 = f(t_n,       y_n)
 *   k2 = f(t_n + h/2, y_n + (h/2) k1)
 *   k3 = f(t_n + h/2, y_n + (h/2) k2)
 *   k4 = f(t_n + h,   y_n + h k3)
 *   y_{n+1} = y_n + (h/6)(k1 + 2 k2 + 2 k3 + k4)
 * Order 4. Local error O(h^5), global error O(h^4).
 * Stability: real-axis limit h*|lambda| <= 2.785, imaginary-axis limit 2.828.
 * Not symplectic: energy of a conservative system drifts slowly (O(h^5) per
 * step for oscillators), monotonically rather than oscillating.
 */
export const rk4 = {
  id: 'rk4',
  name: 'Runge-Kutta 4',
  order: 4,
  symplectic: false,
  requiresSeparable: false,
  evaluationsPerStep: 4,
  stabilityLimit: 2.785,
  notes: 'Explicit, fourth order. Default for dissipative systems. Energy drifts slowly and steadily in conservative systems.',
  createWork: (n) => ({
    k1: new Float64Array(n), k2: new Float64Array(n),
    k3: new Float64Array(n), k4: new Float64Array(n), tmp: new Float64Array(n),
  }),
  step(sys, p, t, y, h, out, w) {
    const n = y.length, { k1, k2, k3, k4, tmp } = w, hh = 0.5 * h, h6 = h / 6;
    sys.rhs(t, y, k1, p);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hh * k1[i];
    sys.rhs(t + hh, tmp, k2, p);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hh * k2[i];
    sys.rhs(t + hh, tmp, k3, p);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + h * k3[i];
    sys.rhs(t + h, tmp, k4, p);
    for (let i = 0; i < n; i++) out[i] = y[i] + h6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  },
};

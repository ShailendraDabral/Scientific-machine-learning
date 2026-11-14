/**
 * Stormer-Verlet in velocity (kick-drift-kick) form.
 * For separable systems  q' = v,  v' = a(t, q)  with state y = [q, v]:
 *   v_{n+1/2} = v_n + (h/2) a(t_n, q_n)
 *   q_{n+1}   = q_n + h v_{n+1/2}
 *   v_{n+1}   = v_{n+1/2} + (h/2) a(t_{n+1}, q_{n+1})
 * Order 2, symplectic and time-reversible. Energy error is bounded, O(h^2),
 * with no secular drift; linear and angular momentum are conserved to
 * round-off for pairwise central forces.
 * Linear stability for an oscillator of frequency w: h*w < 2.
 */
export const verlet = {
  id: 'verlet',
  name: 'Stormer-Verlet',
  order: 2,
  symplectic: true,
  requiresSeparable: true,
  evaluationsPerStep: 2,
  stabilityLimit: 2.0,
  notes: 'Second order, symplectic, time-reversible. Requires acceleration that depends on position and time only. Bounded O(h^2) energy error.',
  createWork: (n) => ({ a: new Float64Array(n / 2) }),
  step(sys, p, t, y, h, out, w) {
    const nq = y.length / 2, a = w.a, hh = 0.5 * h;
    sys.separable.accel(t, y, a, p);
    for (let i = 0; i < nq; i++) {
      out[nq + i] = y[nq + i] + hh * a[i];
      out[i] = y[i] + h * out[nq + i];
    }
    sys.separable.accel(t + h, out, a, p); // reads positions out[0..nq) only
    for (let i = 0; i < nq; i++) out[nq + i] += hh * a[i];
  },
};

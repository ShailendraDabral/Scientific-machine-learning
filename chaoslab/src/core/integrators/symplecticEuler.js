/**
 * Symplectic (semi-implicit) Euler, kick-then-drift variant.
 * For separable systems  q' = v,  v' = a(t, q)  with state y = [q, v]:
 *   v_{n+1} = v_n + h a(t_n, q_n)
 *   q_{n+1} = q_n + h v_{n+1}
 * Order 1, but the map is symplectic: phase-space area is preserved exactly
 * and the energy error stays bounded, O(h), instead of growing.
 * Linear stability for an oscillator of frequency w: h*w < 2.
 */
export const symplecticEuler = {
  id: 'symplecticEuler',
  name: 'Symplectic Euler',
  order: 1,
  symplectic: true,
  requiresSeparable: true,
  evaluationsPerStep: 1,
  stabilityLimit: 2.0,
  notes: 'First order, symplectic. Requires acceleration that depends on position and time only. Bounded O(h) energy error.',
  createWork: (n) => ({ a: new Float64Array(n / 2) }),
  step(sys, p, t, y, h, out, w) {
    const nq = y.length / 2, a = w.a;
    sys.separable.accel(t, y, a, p);
    for (let i = 0; i < nq; i++) {
      const v = y[nq + i] + h * a[i];
      out[nq + i] = v;
      out[i] = y[i] + h * v;
    }
  },
};

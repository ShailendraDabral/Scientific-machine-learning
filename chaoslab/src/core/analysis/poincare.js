import { createStepper } from '../solve.js';

/**
 * Section definitions, by system id. A surface section records the state
 * whenever g(t, y, p) crosses zero upward (and `accept` holds); a strobe
 * section records the state once per drive period.
 */
export const sections = {
  lorenz: {
    kind: 'surface', label: 'plane z = ρ − 1, crossed upward', axes: ['x', 'y'], pick: [0, 1],
    g: (t, y, p) => y[2] - (p.rho - 1),
  },
  rossler: {
    kind: 'surface', label: 'half-plane y = 0 with x < 0', axes: ['x', 'z'], pick: [0, 2],
    g: (t, y) => -y[1], accept: (y) => y[0] < 0,
  },
  duffing: {
    kind: 'strobe', label: 'stroboscopic, t = nT with T = 2π/ω', axes: ['x', 'v'], pick: [0, 1],
    period: (p) => (p.omega > 0 ? (2 * Math.PI) / p.omega : null),
  },
  doublePendulum: {
    kind: 'surface', label: 'θ₂ = 0 (mod 2π) with ω₂ > 0', axes: ['θ₁ (rad, wrapped)', 'ω₁ (rad/s)'], pick: [0, 2],
    g: (t, y) => Math.sin(y[1]), accept: (y) => Math.cos(y[1]) > 0 && y[3] > 0,
    wrap: [0],
  },
};

/**
 * Poincare section of a trajectory. Crossings are located on the cubic
 * Hermite interpolant of the step (which uses the vector field at both
 * ends) by bisection, so their accuracy is O(dt^4) rather than O(dt).
 */
export function poincareSection(system, { params, y0, method = 'rk4', dt, transient = 0, duration, maxPoints = 4000 }) {
  const sec = sections[system.id];
  if (!sec) return { applicable: false, reason: `no section is defined for ${system.name}` };
  const st = createStepper(system, { params, y0, method, dt });
  const p = st.params, n = st.y.length;
  const period = sec.kind === 'strobe' ? sec.period(p) : null;
  if (sec.kind === 'strobe' && !period) return { applicable: false, reason: 'the drive frequency is zero: there is no period to strobe at' };
  st.advance(Math.round(transient / dt));
  const steps = Math.round(duration / dt);
  const y0v = new Float64Array(n), f0 = new Float64Array(n), f1 = new Float64Array(n), out = new Float64Array(n);
  const hermite = (s, y1) => {
    const s2 = s * s, s3 = s2 * s;
    const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    for (let i = 0; i < n; i++) out[i] = h00 * y0v[i] + h10 * dt * f0[i] + h01 * y1[i] + h11 * dt * f1[i];
    return out;
  };
  const a = [], b = [];
  let residual = 0;
  let gPrev = sec.kind === 'surface' ? sec.g(st.t, st.y, p) : 0;
  for (let k = 0; k < steps && a.length < maxPoints; k++) {
    y0v.set(st.y);
    const tPrev = st.t;
    st.advance(1);
    const y1 = st.y;
    let s = null;
    if (sec.kind === 'strobe') {
      const m = Math.floor(st.t / period);
      if (m > Math.floor(tPrev / period)) s = (m * period - tPrev) / dt;
    } else {
      const gNow = sec.g(st.t, y1, p);
      if (gPrev < 0 && gNow >= 0 && (!sec.accept || sec.accept(y1))) s = -1; // bracketed: refine below
      gPrev = gNow;
    }
    if (s === null) continue;
    system.rhs(tPrev, y0v, f0, p);
    system.rhs(st.t, y1, f1, p);
    if (s === -1) {
      let lo = 0, hi = 1;
      for (let it = 0; it < 50; it++) {
        const mid = 0.5 * (lo + hi);
        if (sec.g(tPrev + mid * dt, hermite(mid, y1), p) < 0) lo = mid; else hi = mid;
      }
      s = 0.5 * (lo + hi);
      residual = Math.max(residual, Math.abs(sec.g(tPrev + s * dt, hermite(s, y1), p)));
    }
    const pt = hermite(s, y1);
    let u = pt[sec.pick[0]], v = pt[sec.pick[1]];
    if (sec.wrap?.includes(sec.pick[0])) u -= 2 * Math.PI * Math.round(u / (2 * Math.PI));
    a.push(u); b.push(v);
  }
  return {
    applicable: true, a, b, axes: sec.axes, label: sec.label, kind: sec.kind,
    points: a.length, residual, duration: steps * dt, transient: Math.round(transient / dt) * dt, method, dt,
  };
}

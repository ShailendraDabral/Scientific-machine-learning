import { Trail, boxFrame, sphere, COLORS } from '../render/scene.js';
import { line, PLOT_COLORS as C } from '../ui/plots.js';
import { LongSeries } from '../ui/history.js';
import { fmt } from '../ui/dom.js';
import { lorenz as core } from '../core/index.js';

const S = 1 / 22, Z0 = 25; // scene scale and vertical centre
const sep = (d, o) => Math.hypot(d[o + 3] - d[o], d[o + 4] - d[o + 1], d[o + 5] - d[o + 2]);

export default {
  id: 'lorenz', title: 'Lorenz attractor', group: 'Dissipative flows', system: 'lorenz',
  tex: [String.raw`\begin{aligned}\dot x &= \sigma (y - x)\\ \dot y &= x(\rho - z) - y\\ \dot z &= xy - \beta z\end{aligned}`],
  blurb: 'Two trajectories start a distance δ₀ apart in x. Their separation grows as e^{λt} until it reaches the size of the attractor; the fitted rate λ is the largest Lyapunov exponent.',
  params: [
    { key: 'sigma', tex: '\\sigma', value: 10, min: 0.1, max: 30, step: 0.1 },
    { key: 'rho', tex: '\\rho', value: 28, min: 0, max: 120, step: 0.1 },
    { key: 'beta', tex: '\\beta', value: 8 / 3, min: 0.1, max: 10, step: 0.01 },
  ],
  ic: [
    { key: 'x0', tex: 'x_0', value: 5, min: -20, max: 20, step: 0.1 },
    { key: 'y0', tex: 'y_0', value: 5, min: -25, max: 25, step: 0.1 },
    { key: 'z0', tex: 'z_0', value: 5, min: 0, max: 50, step: 0.1 },
    { key: 'logd', tex: '\\log_{10}\\delta_0', value: -6, min: -12, max: -1, step: 0.5, liveOnly: true },
  ],
  numerics: { dt: 0.005, dtMin: 1e-4, dtMax: 0.1, method: 'rk4', methods: ['euler', 'midpoint', 'rk4'], rate: 2 },
  presets: [
    { name: 'ρ = 28 · butterfly attractor', p: { sigma: 10, rho: 28, beta: 8 / 3 } },
    { name: 'ρ = 14 · decays to an equilibrium', p: { sigma: 10, rho: 14, beta: 8 / 3 } },
    { name: 'ρ = 100 · periodic window', p: { sigma: 10, rho: 100, beta: 8 / 3 } },
  ],
  legend: [['#E8A33D', 'trajectory A'], ['#4CC9F0', 'trajectory B = A + δ₀'], ['#7BD88F', 'equilibria C±']],

  coreParams: (p) => ({ sigma: p.sigma, rho: p.rho, beta: p.beta }),
  initialStates: (p, ic) => [[ic.x0, ic.y0, ic.z0], [ic.x0 + 10 ** ic.logd, ic.y0, ic.z0]],

  createView(sm, lab) {
    const a = new Trail(24000, COLORS.amber).addTo(sm.scene);
    const b = new Trail(24000, COLORS.cyan, 0.75).addTo(sm.scene);
    sm.scene.add(boxFrame([-1, -Z0 * S, -1.2], [1, Z0 * S, 1.2]));
    const eq = [sphere(COLORS.green, 0.014), sphere(COLORS.green, 0.014)];
    sm.scene.add(...eq);
    sm.setCamera([1.1, 0.7, 3.7]);
    const view = {
      reset() { a.clear(); b.clear(); view.params(lab.p); },
      params(p) {
        const show = p.rho > 1, s = show ? Math.sqrt(p.beta * (p.rho - 1)) : 0;
        eq.forEach((m, i) => {
          m.visible = show;
          const sg = i ? -1 : 1;
          m.position.set(sg * s * S, (p.rho - 1 - Z0) * S, sg * s * S);
        });
      },
      append(rows, d) {
        for (let r = 0; r < rows; r++) {
          const o = r * 6;
          a.push(d[o] * S, (d[o + 2] - Z0) * S, d[o + 1] * S);
          b.push(d[o + 3] * S, (d[o + 5] - Z0) * S, d[o + 4] * S);
        }
        a.commit(); b.commit();
      },
    };
    return view;
  },

  onReset(lab) {
    lab.store.sep = new LongSeries(2);
    // least-squares fit of ln|separation| against t during the growth phase
    lab.store.fit = { n: 0, st: 0, sy: 0, stt: 0, sty: 0, lo: Infinity, hi: -Infinity, saturatedAt: null };
  },

  onSamples(lab, res) {
    const f = lab.store.fit;
    for (let r = 0; r < res.rows; r++) {
      const s = sep(res.data, r * 6), t = res.times[r];
      if (s > 0) lab.store.sep.push(t, Math.log10(s));
      if (f.saturatedAt === null && s > 0) {
        if (s > 1) f.saturatedAt = t;
        else { const y = Math.log(s); f.n++; f.lo = Math.min(f.lo, y); f.hi = Math.max(f.hi, y); f.st += t; f.sy += y; f.stt += t * t; f.sty += t * y; }
      }
    }
  },

  plots: [
    {
      title: 'x(t) — both trajectories', x: 't', y: 'x',
      traces(lab) {
        const A = lab.hist.xy((t) => t, (t, d, o) => d[o], { window: 30 });
        const B = lab.hist.xy((t) => t, (t, d, o) => d[o + 3], { window: 30 });
        return [line(A.x, A.y, 'A', C.amber), line(B.x, B.y, 'B', C.cyan, { line: { color: C.cyan, width: 1.2, dash: 'dot' } })];
      },
    },
    {
      title: 'Phase plane x–z', x: 'x', y: 'z',
      traces(lab) {
        const A = lab.hist.xy((t, d, o) => d[o], (t, d, o) => d[o + 2], { window: 30, max: 3000 });
        const B = lab.hist.xy((t, d, o) => d[o + 3], (t, d, o) => d[o + 5], { window: 30, max: 3000 });
        return [line(A.x, A.y, 'A', C.amber, { line: { color: C.amber, width: 0.8 } }),
          line(B.x, B.y, 'B', C.cyan, { line: { color: C.cyan, width: 0.8 } })];
      },
    },
    {
      title: 'Separation |B − A| since reset', x: 't', y: 'log₁₀ |Δ|',
      traces(lab) {
        const s = lab.store.sep, f = lab.store.fit, out = [line(s.col(0), s.col(1), 'measured', C.amber)];
        const fit = slope(f);
        if (fit) {
          const t1 = f.saturatedAt ?? lab.t;
          out.push(line([0, t1], [fit.c / Math.LN10, (fit.c + fit.m * t1) / Math.LN10], `fit, λ = ${fit.m.toFixed(3)}`, C.green,
            { line: { color: C.green, width: 1, dash: 'dash' } }));
        }
        return out;
      },
    },
  ],

  readouts(lab) {
    const a = lab.state(0), b = lab.state(1), f = lab.store.fit, fit = slope(f);
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    return [
      ['A (x, y, z)', `${fmt(a[0], 4)}, ${fmt(a[1], 4)}, ${fmt(a[2], 4)}`],
      ['B (x, y, z)', `${fmt(b[0], 4)}, ${fmt(b[1], 4)}, ${fmt(b[2], 4)}`],
      ['|B − A|', fmt(d, 4)],
      ['growth factor |Δ| / δ₀', fmt(d / 10 ** lab.ic.logd, 4)],
      ['λ fitted from ln|Δ|', fit ? fmt(fit.m, 4) : 'waiting for 100× growth'],
      ['separation reached 1 at t', f.saturatedAt === null ? 'not yet' : fmt(f.saturatedAt, 4)],
      ['volume contraction ∇·f', fmt(core.divergence(lab.p), 5)],
    ];
  },
};

function slope(f) {
  // wait until the separation has grown a hundredfold, so the slope is meaningful
  if (f.n < 40 || f.hi - f.lo < Math.log(100)) return null;
  const den = f.n * f.stt - f.st * f.st;
  if (den <= 0) return null;
  const m = (f.n * f.sty - f.st * f.sy) / den;
  return { m, c: (f.sy - m * f.st) / f.n };
}

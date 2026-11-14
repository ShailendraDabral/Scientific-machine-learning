import { Trail, PointSet, segments, COLORS } from '../render/scene.js';
import { line, dots, PLOT_COLORS as C } from '../ui/plots.js';
import { fmt } from '../ui/dom.js';
import { duffing as core } from '../core/index.js';

const S = 0.75;

export default {
  id: 'duffing', title: 'Duffing oscillator', group: 'Driven oscillator', system: 'duffing',
  tex: [String.raw`\ddot x + \delta \dot x + \alpha x + \beta x^3 = \gamma \cos(\omega t)`,
    String.raw`V(x) = \tfrac12 \alpha x^2 + \tfrac14 \beta x^4`],
  blurb: 'The Poincaré section records (x, v) once per drive period T = 2π/ω, interpolated inside the step that crosses t = nT. A few points mean periodic motion; a folded fractal set means chaos. Symplectic methods are available only when δ = 0.',
  params: [
    { key: 'alpha', tex: '\\alpha', value: -1, min: -2, max: 2, step: 0.01 },
    { key: 'beta', tex: '\\beta', value: 1, min: 0, max: 2, step: 0.01 },
    { key: 'delta', tex: '\\delta', value: 0.3, min: 0, max: 1, step: 0.005 },
    { key: 'gamma', tex: '\\gamma', value: 0.5, min: 0, max: 1.5, step: 0.005 },
    { key: 'omega', tex: '\\omega', value: 1.2, min: 0.1, max: 3, step: 0.01 },
  ],
  ic: [
    { key: 'x0', tex: 'x_0', value: 1, min: -2, max: 2, step: 0.01 },
    { key: 'v0', tex: 'v_0', value: 0, min: -2, max: 2, step: 0.01 },
  ],
  numerics: { dt: 0.005, dtMin: 1e-4, dtMax: 0.2, method: 'rk4',
    methods: ['euler', 'midpoint', 'rk4', 'symplecticEuler', 'verlet'], rate: 6 },
  presets: [
    { name: 'γ = 0.50 · chaotic', p: { alpha: -1, beta: 1, delta: 0.3, gamma: 0.5, omega: 1.2 }, method: 'rk4' },
    { name: 'γ = 0.20', p: { alpha: -1, beta: 1, delta: 0.3, gamma: 0.2, omega: 1.2 }, method: 'rk4' },
    { name: 'γ = 0.28', p: { alpha: -1, beta: 1, delta: 0.3, gamma: 0.28, omega: 1.2 }, method: 'rk4' },
    { name: 'γ = 0.37', p: { alpha: -1, beta: 1, delta: 0.3, gamma: 0.37, omega: 1.2 }, method: 'rk4' },
    { name: 'δ = 0, γ = 0 · conservative (Verlet)', p: { alpha: -1, beta: 1, delta: 0, gamma: 0, omega: 1.2 },
      ic: { x0: 0.5, v0: 0.8 }, method: 'verlet' },
  ],
  legend: [['#E8A33D', 'phase trajectory (x, v)'], ['#4CC9F0', 'stroboscopic section, t = nT']],

  coreParams: (p) => ({ alpha: p.alpha, beta: p.beta, delta: p.delta, gamma: p.gamma, omega: p.omega }),
  initialStates: (p, ic) => [[ic.x0, ic.v0]],
  strobe: (p) => (p.omega > 0 ? (2 * Math.PI) / p.omega : null),

  createView(sm) {
    const tr = new Trail(16000, COLORS.amber, 0.7).addTo(sm.scene);
    const pts = new PointSet(6000, COLORS.cyan, 3.5);
    sm.scene.add(pts.object, segments([-1.6, 0, 0, 1.6, 0, 0, 0, -1.2, 0, 0, 1.2, 0], COLORS.line));
    sm.setCamera([0, 0, 4.2]);
    return {
      reset() { tr.clear(); pts.clear(); },
      params() { pts.clear(); },
      append(rows, d) {
        for (let r = 0; r < rows; r++) tr.push(d[r * 2] * S, d[r * 2 + 1] * S, 0);
        tr.commit();
      },
      section(points) { for (const p of points) pts.push(p[0] * S, p[1] * S, 0.001); },
    };
  },

  // A parameter change moves the attractor: start the section afresh.
  onParams(lab) { lab.section = []; },

  plots: [
    {
      title: 'x(t)', x: 't', y: 'x',
      traces(lab) {
        const X = lab.hist.xy((t) => t, (t, d, o) => d[o], { window: 80 });
        return [line(X.x, X.y, 'x', C.amber)];
      },
    },
    {
      title: 'Potential V(x) and current position', x: 'x', y: 'V',
      traces(lab) {
        const xs = [], vs = [], y = lab.state(0);
        const span = Math.max(1.8, Math.abs(y[0]) * 1.2);
        for (let i = 0; i <= 120; i++) { const x = -span + (2 * span * i) / 120; xs.push(x); vs.push(core.potential(x, lab.p)); }
        return [line(xs, vs, 'V(x)', C.grey), dots([y[0]], [core.potential(y[0], lab.p)], 'oscillator', C.amber, 9)];
      },
    },
    {
      title: 'Poincaré section, t = nT', x: 'x', y: 'v',
      traces(lab) {
        const s = lab.section;
        return [dots(s.map((p) => p[0]), s.map((p) => p[1]), `${s.length} points`, C.cyan, 3)];
      },
    },
  ],

  readouts(lab) {
    const y = lab.state(0), p = lab.p, inv = lab.reports[0]?.invariants.energy;
    const rows = [
      ['x, v', `${fmt(y[0], 4)}, ${fmt(y[1], 4)}`],
      ['drive period T = 2π/ω', fmt((2 * Math.PI) / p.omega, 5)],
      ['steps per period T/dt', fmt((2 * Math.PI) / p.omega / lab.dt, 5)],
      ['mechanical energy ½v² + V', fmt(0.5 * y[1] * y[1] + core.potential(y[0], p), 5)],
      ['section points', String(lab.section.length)],
    ];
    if (inv) rows.push(['energy drift (δ = γ = 0)', fmt(inv.maxRelDrift ?? inv.maxAbsDrift, 3)]);
    return rows;
  },
};

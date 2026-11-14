import { Trail, boxFrame, sphere, COLORS } from '../render/scene.js';
import { line, dots, PLOT_COLORS as C } from '../ui/plots.js';
import { fmt } from '../ui/dom.js';
import { rossler as core } from '../core/index.js';

const S = 1 / 11;

export default {
  id: 'rossler', title: 'Rössler attractor', group: 'Dissipative flows', system: 'rossler',
  tex: [String.raw`\begin{aligned}\dot x &= -y - z\\ \dot y &= x + a y\\ \dot z &= b + z(x - c)\end{aligned}`],
  blurb: 'The return map plots each local maximum of x against the next one, taken from the simulated trajectory. A single smooth hump is the signature of low-dimensional chaos.',
  params: [
    { key: 'a', tex: 'a', value: 0.2, min: 0.05, max: 0.4, step: 0.005 },
    { key: 'b', tex: 'b', value: 0.2, min: 0.05, max: 2, step: 0.01 },
    { key: 'c', tex: 'c', value: 5.7, min: 1, max: 18, step: 0.05 },
  ],
  ic: [
    { key: 'x0', tex: 'x_0', value: 1, min: -10, max: 10, step: 0.1 },
    { key: 'y0', tex: 'y_0', value: 1, min: -10, max: 10, step: 0.1 },
    { key: 'z0', tex: 'z_0', value: 0, min: 0, max: 10, step: 0.1 },
  ],
  numerics: { dt: 0.01, dtMin: 1e-4, dtMax: 0.2, method: 'rk4', methods: ['euler', 'midpoint', 'rk4'], rate: 8 },
  presets: [
    { name: 'c = 5.7 · chaotic attractor', p: { a: 0.2, b: 0.2, c: 5.7 } },
    { name: 'c = 2.5', p: { a: 0.2, b: 0.2, c: 2.5 } },
    { name: 'c = 3.5', p: { a: 0.2, b: 0.2, c: 3.5 } },
    { name: 'c = 4.0', p: { a: 0.2, b: 0.2, c: 4 } },
  ],
  legend: [['#E8A33D', 'trajectory'], ['#7BD88F', 'equilibria']],

  coreParams: (p) => ({ a: p.a, b: p.b, c: p.c }),
  initialStates: (p, ic) => [[ic.x0, ic.y0, ic.z0]],

  createView(sm, lab) {
    const tr = new Trail(24000, COLORS.amber).addTo(sm.scene);
    sm.scene.add(boxFrame([-1.2, -0.6, -1.2], [1.2, 1.6, 1.2]));
    const eq = [sphere(COLORS.green, 0.014), sphere(COLORS.green, 0.014)];
    sm.scene.add(...eq);
    sm.setCamera([2.6, 2.2, 3.4], [0, 0.3, 0]);
    const view = {
      reset() { tr.clear(); view.params(lab.p); },
      params(p) {
        let pts = [];
        try { pts = core.equilibria(core.resolveParams(lab.exp.coreParams(p)).params); } catch { /* invalid: hide */ }
        eq.forEach((m, i) => {
          m.visible = !!pts[i] && Math.abs(pts[i][2]) < 40;
          if (pts[i]) m.position.set(pts[i][0] * S, pts[i][2] * S - 0.6, pts[i][1] * S);
        });
      },
      append(rows, d) {
        for (let r = 0; r < rows; r++) tr.push(d[r * 3] * S, d[r * 3 + 2] * S - 0.6, d[r * 3 + 1] * S);
        tr.commit();
      },
    };
    return view;
  },

  onReset(lab) { lab.store.maxima = []; lab.store.prev = null; },

  // Local maxima of x from consecutive samples (parabolic refinement of the peak value).
  onSamples(lab, res) {
    const st = lab.store;
    for (let r = 0; r < res.rows; r++) {
      const x = res.data[r * 3];
      if (st.prev && st.prev.length === 2) {
        const [x0, x1] = st.prev;
        if (x1 > x0 && x1 >= x && res.times[r] > 20) {
          const den = x0 - 2 * x1 + x;
          st.maxima.push(den < 0 ? x1 - ((x0 - x) ** 2) / (8 * den) : x1);
          if (st.maxima.length > 1500) st.maxima.shift();
        }
        st.prev = [x1, x];
      } else st.prev = st.prev ? [st.prev[0], x] : [x];
    }
  },

  plots: [
    {
      title: 'x(t) and z(t)', x: 't', y: 'x, z',
      traces(lab) {
        const X = lab.hist.xy((t) => t, (t, d, o) => d[o], { window: 120 });
        const Z = lab.hist.xy((t) => t, (t, d, o) => d[o + 2], { window: 120 });
        return [line(X.x, X.y, 'x', C.amber), line(Z.x, Z.y, 'z', C.cyan)];
      },
    },
    {
      title: 'Phase plane x–y', x: 'x', y: 'y',
      traces(lab) {
        const A = lab.hist.xy((t, d, o) => d[o], (t, d, o) => d[o + 1], { max: 4000 });
        return [line(A.x, A.y, 'trajectory', C.amber, { line: { color: C.amber, width: 0.8 } })];
      },
    },
    {
      title: 'Return map of x maxima', x: 'xₙ(max)', y: 'xₙ₊₁(max)',
      traces(lab) {
        const m = lab.store.maxima;
        return [dots(m.slice(0, -1), m.slice(1), 'successive maxima', C.amber, 3)];
      },
    },
  ],

  readouts(lab) {
    const y = lab.state(0);
    return [
      ['x, y, z', `${fmt(y[0], 4)}, ${fmt(y[1], 4)}, ${fmt(y[2], 4)}`],
      ['local contraction a + x − c', fmt(lab.p.a + y[0] - lab.p.c, 4)],
      ['x maxima recorded (t > 20)', String(lab.store.maxima.length)],
    ];
  },
};

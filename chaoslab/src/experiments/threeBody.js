import { Trail, sphere, COLORS } from '../render/scene.js';
import { line, PLOT_COLORS as C } from '../ui/plots.js';
import { LongSeries } from '../ui/history.js';
import { fmt } from '../ui/dom.js';
import { threeBody as core, threeBodyPresets, toCenterOfMassFrame } from '../core/index.js';

const BODY = [COLORS.amber, COLORS.cyan, COLORS.rose];
const BODY_CSS = [C.amber, C.cyan, C.rose];

/** Inspector values for a core preset { params, y0 }. */
function fromPreset(ps, extra = {}) {
  const y = ps.y0, ic = {};
  for (let i = 0; i < 3; i++) {
    ic[`x${i + 1}`] = y[2 * i]; ic[`y${i + 1}`] = y[2 * i + 1];
    ic[`vx${i + 1}`] = y[6 + 2 * i]; ic[`vy${i + 1}`] = y[6 + 2 * i + 1];
  }
  const [m1, m2, m3] = ps.params.masses;
  return { p: { m1, m2, m3, eps: 0, ...extra.p }, ic, ...extra.rest };
}
const fig8 = fromPreset(threeBodyPresets.figureEight());

const icFields = [];
for (let i = 1; i <= 3; i++) {
  for (const [k, t] of [['x', `x_${i}`], ['y', `y_${i}`], ['vx', `v_{x${i}}`], ['vy', `v_{y${i}}`]]) {
    const key = `${k}${i}`;
    icFields.push({ key, tex: t, value: fig8.ic[key], slider: false, min: -1e3, max: 1e3, aria: key });
  }
}

export default {
  id: 'three-body', title: 'Three-body problem', group: 'Hamiltonian systems', system: 'threeBody',
  tex: [String.raw`\ddot{\mathbf r}_i = G\sum_{j\neq i} m_j\,\frac{\mathbf r_j-\mathbf r_i}{(|\mathbf r_j-\mathbf r_i|^2+\varepsilon^2)^{3/2}}`,
    String.raw`E=\sum_i \tfrac12 m_i|\mathbf v_i|^2-G\sum_{i<j}\frac{m_i m_j}{r_{ij}},\quad L_z=\sum_i m_i(x_i v_{y,i}-y_i v_{x,i})`],
  blurb: 'G = 1. Energy and angular momentum are conserved by the model; the lower-middle plot shows how well the chosen integrator keeps them. Störmer–Verlet holds the energy error bounded; Forward Euler does not.',
  icNote: 'Positions and velocities are shifted to the centre-of-mass frame at reset, so the system does not drift out of view.',
  params: [
    { key: 'm1', tex: 'm_1', value: 1, min: 0.1, max: 10, step: 0.05 },
    { key: 'm2', tex: 'm_2', value: 1, min: 0.1, max: 10, step: 0.05 },
    { key: 'm3', tex: 'm_3', value: 1, min: 0.1, max: 10, step: 0.05 },
    { key: 'eps', tex: '\\varepsilon', value: 0, min: 0, max: 0.2, step: 0.005 },
  ],
  ic: icFields,
  numerics: { dt: 0.001, dtMin: 1e-5, dtMax: 0.05, method: 'verlet',
    methods: ['euler', 'midpoint', 'rk4', 'symplecticEuler', 'verlet'], rate: 1 },
  presets: [
    { name: 'Figure-eight choreography', ...fig8, dt: 0.001, method: 'verlet' },
    { name: 'Lagrange equilateral triangle', ...fromPreset(threeBodyPresets.lagrange()), dt: 0.001, method: 'verlet' },
    { name: 'Pythagorean 3–4–5 (softened, ε = 0.02)', ...fromPreset(threeBodyPresets.pythagorean(), { p: { eps: 0.02 } }),
      dt: 0.0002, method: 'verlet' },
  ],
  legend: [['#E8A33D', 'body 1'], ['#4CC9F0', 'body 2'], ['#F06292', 'body 3']],
  paramsRestart: true, // changing a mass redefines the conserved quantities: restart the run

  coreParams: (p) => ({ G: 1, softening: p.eps, dim: 2, masses: [p.m1, p.m2, p.m3] }),
  initialStates(p, ic) {
    const y = new Float64Array(12);
    for (let i = 0; i < 3; i++) {
      y[2 * i] = ic[`x${i + 1}`]; y[2 * i + 1] = ic[`y${i + 1}`];
      y[6 + 2 * i] = ic[`vx${i + 1}`]; y[6 + 2 * i + 1] = ic[`vy${i + 1}`];
    }
    const { params } = core.resolveParams(this.coreParams(p)); // throws on invalid masses
    return [toCenterOfMassFrame(y, params)];
  },

  createView(sm, lab) {
    const trails = BODY.map((c) => new Trail(12000, c, 0.8).addTo(sm.scene));
    const bodies = BODY.map((c) => sphere(c, 0.03));
    trails.forEach((t) => { t.head.visible = false; });
    sm.scene.add(...bodies);
    sm.setCamera([0, 0, 3.6]);
    let s = 1;
    return {
      reset(y) {
        let ext = 0;
        for (let i = 0; i < 6; i++) ext = Math.max(ext, Math.abs(y[i]));
        s = 1 / (1.25 * (ext || 1));
        trails.forEach((t) => t.clear());
        bodies.forEach((b, i) => b.scale.setScalar(0.03 * Math.cbrt(lab.p[`m${i + 1}`])));
      },
      append(rows, d) {
        for (let r = 0; r < rows; r++) {
          for (let i = 0; i < 3; i++) trails[i].push(d[r * 12 + 2 * i] * s, d[r * 12 + 2 * i + 1] * s, 0);
        }
        trails.forEach((t) => { t.commit(); t.head.visible = false; });
      },
      state(y) { bodies.forEach((b, i) => b.position.set(y[2 * i] * s, y[2 * i + 1] * s, 0)); },
    };
  },

  onReset(lab) {
    const { params } = core.resolveParams(this.coreParams(lab.p));
    const inv = core.invariants(params);
    const y = lab.state(0);
    lab.store.params = params;
    lab.store.inv = inv;
    lab.store.E0 = inv.energy(0, y);
    lab.store.L0 = inv.angularMomentumZ(0, y);
    lab.store.cons = new LongSeries(3);
    lab.view.state(y);
  },

  onSamples(lab, res) {
    const { inv, E0, L0, cons } = lab.store, row = new Float64Array(12);
    for (let r = 0; r < res.rows; r++) {
      for (let i = 0; i < 12; i++) row[i] = res.data[r * 12 + i];
      const dE = Math.abs(inv.energy(0, row) - E0) / (Math.abs(E0) || 1);
      const dL = Math.abs(inv.angularMomentumZ(0, row) - L0);
      cons.push(res.times[r], Math.max(dE, 1e-17), Math.max(dL, 1e-17));
    }
  },

  plots: [
    {
      title: 'Energy: kinetic, potential, total', x: 't', y: 'energy',
      traces(lab) {
        const { params, inv } = lab.store, row = new Float64Array(12), o = { window: 25 };
        const kin = (d, i) => {
          let k = 0;
          for (let b = 0; b < 3; b++) k += 0.5 * params.masses[b] * (d[i + 6 + 2 * b] ** 2 + d[i + 7 + 2 * b] ** 2);
          return k;
        };
        const tot = (d, i) => { for (let j = 0; j < 12; j++) row[j] = d[i + j]; return inv.energy(0, row); };
        const K = lab.hist.xy((t) => t, (t, d, i) => kin(d, i), o);
        const E = lab.hist.xy((t) => t, (t, d, i) => tot(d, i), o);
        const U = { x: E.x, y: E.y.map((e, j) => e - K.y[j]) };
        return [line(K.x, K.y, 'K', C.rose), line(U.x, U.y, 'U', C.cyan),
          line(E.x, E.y, 'E = K + U', C.green, { line: { color: C.green, width: 1.8 } })];
      },
    },
    {
      title: 'Conservation error since reset', x: 't', y: 'error',
      layout: { yaxis: { type: 'log', exponentformat: 'power', gridcolor: '#1B222C', linecolor: '#2A3340',
        tickfont: { size: 10 }, title: { text: 'error', font: { size: 11 } }, automargin: true } },
      traces(lab) {
        const c = lab.store.cons;
        return [line(c.col(0), c.col(1), '|E − E₀| / |E₀|', C.green), line(c.col(0), c.col(2), '|L_z − L₀|', C.amber)];
      },
    },
    {
      title: 'Phase plane (x, vₓ) of each body', x: 'x', y: 'vₓ',
      traces(lab) {
        return [0, 1, 2].map((b) => {
          const A = lab.hist.xy((t, d, o) => d[o + 2 * b], (t, d, o) => d[o + 6 + 2 * b], { window: 25, max: 2500 });
          return line(A.x, A.y, `body ${b + 1}`, BODY_CSS[b], { line: { color: BODY_CSS[b], width: 0.9 } });
        });
      },
    },
  ],

  readouts(lab) {
    const y = lab.state(0), { inv, E0, L0 } = lab.store;
    const E = inv.energy(0, y), L = inv.angularMomentumZ(0, y);
    let rmin = Infinity;
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) rmin = Math.min(rmin, Math.hypot(y[2 * j] - y[2 * i], y[2 * j + 1] - y[2 * i + 1]));
    return [
      ['total energy E', fmt(E, 8)],
      ['|E − E₀| / |E₀|', fmt(Math.abs(E - E0) / (Math.abs(E0) || 1), 3)],
      ['angular momentum L_z', fmt(L, 8)],
      ['|L_z − L₀|', fmt(Math.abs(L - L0), 3)],
      ['|P| (total momentum)', fmt(Math.hypot(inv.momentumX(0, y), inv.momentumY(0, y)), 3)],
      ['closest pair distance', fmt(rmin, 4)],
    ];
  },
};

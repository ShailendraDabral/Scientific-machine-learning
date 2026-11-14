import * as THREE from 'three';
import { Trail, segments, sphere, COLORS } from '../render/scene.js';
import { line, dots, PLOT_COLORS as C } from '../ui/plots.js';
import { fmt } from '../ui/dom.js';

const DEG = Math.PI / 180;
const PIVOT_Y = 0.25;
const wrap = (a) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

// Energies evaluated on a stored row (same expressions as the core model).
function potential(d, o, p) {
  return -(p.m1 + p.m2) * p.g * p.l1 * Math.cos(d[o]) - p.m2 * p.g * p.l2 * Math.cos(d[o + 1]);
}
function kinetic(d, o, p) {
  const w1 = d[o + 2], w2 = d[o + 3];
  return 0.5 * (p.m1 + p.m2) * p.l1 * p.l1 * w1 * w1 + 0.5 * p.m2 * p.l2 * p.l2 * w2 * w2
    + p.m2 * p.l1 * p.l2 * w1 * w2 * Math.cos(d[o] - d[o + 1]);
}

export default {
  id: 'double-pendulum', title: 'Double pendulum', group: 'Hamiltonian systems', system: 'doublePendulum',
  tex: [String.raw`\begin{aligned}\dot\omega_1 &= \tfrac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2) - 2m_2\sin\Delta\,(\omega_2^2 l_2+\omega_1^2 l_1\cos\Delta)}{l_1(2m_1+m_2-m_2\cos 2\Delta)}\\[4pt] \dot\omega_2 &= \tfrac{2\sin\Delta\,(\omega_1^2 l_1(m_1+m_2)+g(m_1+m_2)\cos\theta_1+\omega_2^2 l_2 m_2\cos\Delta)}{l_2(2m_1+m_2-m_2\cos 2\Delta)}\end{aligned}`,
    String.raw`\Delta = \theta_1 - \theta_2`],
  blurb: 'The model conserves total energy exactly, so any change in E is integration error. Compare Forward Euler with RK4 at the same step. Symplectic methods are not offered: the Hamiltonian is not separable.',
  params: [
    { key: 'm1', tex: 'm_1', value: 1, min: 0.1, max: 5, step: 0.05, unit: 'kg' },
    { key: 'm2', tex: 'm_2', value: 1, min: 0.1, max: 5, step: 0.05, unit: 'kg' },
    { key: 'l1', tex: 'l_1', value: 1, min: 0.2, max: 2, step: 0.01, unit: 'm' },
    { key: 'l2', tex: 'l_2', value: 1, min: 0.2, max: 2, step: 0.01, unit: 'm' },
    { key: 'g', tex: 'g', value: 9.81, min: 0, max: 25, step: 0.01, unit: 'm/s²' },
  ],
  ic: [
    { key: 'th1', tex: '\\theta_1', value: 120, min: -180, max: 180, step: 0.5, unit: '°' },
    { key: 'th2', tex: '\\theta_2', value: -20, min: -180, max: 180, step: 0.5, unit: '°' },
    { key: 'w1', tex: '\\omega_1', value: 0, min: -10, max: 10, step: 0.05, unit: 'rad/s' },
    { key: 'w2', tex: '\\omega_2', value: 0, min: -10, max: 10, step: 0.05, unit: 'rad/s' },
  ],
  numerics: { dt: 0.002, dtMin: 1e-4, dtMax: 0.05, method: 'rk4', methods: ['euler', 'midpoint', 'rk4'], rate: 1, timeUnit: 's' },
  presets: [
    { name: 'Chaotic release (120°, −20°)', ic: { th1: 120, th2: -20, w1: 0, w2: 0 } },
    { name: 'Small oscillation (10°, 14°)', ic: { th1: 10, th2: 14.14, w1: 0, w2: 0 } },
    { name: 'Both rods horizontal (90°, 90°)', ic: { th1: 90, th2: 90, w1: 0, w2: 0 } },
    { name: 'Fast rotation', ic: { th1: 0, th2: 0, w1: 8, w2: -6 } },
  ],
  legend: [['#E8A33D', 'lower bob path'], ['#4CC9F0', 'upper bob path']],
  paramsRestart: false,

  coreParams: (p) => ({ m1: p.m1, m2: p.m2, l1: p.l1, l2: p.l2, g: p.g }),
  initialStates: (p, ic) => [[ic.th1 * DEG, ic.th2 * DEG, ic.w1, ic.w2]],

  createView(sm, lab) {
    const t2 = new Trail(6000, COLORS.amber, 0.8).addTo(sm.scene);
    const t1 = new Trail(1500, COLORS.cyan, 0.35).addTo(sm.scene);
    t1.head.visible = false; t2.head.visible = false;
    const rodGeo = new THREE.BufferGeometry();
    const rodPos = new Float32Array(9);
    rodGeo.setAttribute('position', new THREE.BufferAttribute(rodPos, 3));
    const rods = new THREE.Line(rodGeo, new THREE.LineBasicMaterial({ color: 0xc9d1db }));
    rods.frustumCulled = false;
    const bob1 = sphere(COLORS.cyan, 0.04), bob2 = sphere(COLORS.amber, 0.04), pivot = sphere(0xc9d1db, 0.014);
    pivot.position.set(0, PIVOT_Y, 0);
    sm.scene.add(rods, bob1, bob2, pivot,
      segments([0, PIVOT_Y, 0, 0, PIVOT_Y - 1.05, 0, -0.25, PIVOT_Y, 0, 0.25, PIVOT_Y, 0], COLORS.line));
    sm.setCamera([0, 0, 3.6]);
    let s = 0.5;
    const pos = (d, o) => {
      const p = lab.p;
      const x1 = s * p.l1 * Math.sin(d[o]), y1 = PIVOT_Y - s * p.l1 * Math.cos(d[o]);
      return [x1, y1, x1 + s * p.l2 * Math.sin(d[o + 1]), y1 - s * p.l2 * Math.cos(d[o + 1])];
    };
    const view = {
      reset() { t1.clear(); t2.clear(); view.params(lab.p); },
      params(p) {
        const ns = 1 / (p.l1 + p.l2);
        if (ns !== s) { s = ns; t1.clear(); t2.clear(); } // lengths changed: old path no longer to scale
        bob1.scale.setScalar(0.035 * Math.cbrt(p.m1));
        bob2.scale.setScalar(0.035 * Math.cbrt(p.m2));
      },
      append(rows, d) {
        for (let r = 0; r < rows; r++) {
          const [x1, y1, x2, y2] = pos(d, r * 4);
          t1.push(x1, y1, 0); t2.push(x2, y2, 0);
        }
        t1.commit(); t2.commit();
        t1.head.visible = false; t2.head.visible = false;
        view.state(lab.states);
      },
      state(y) {
        const [x1, y1, x2, y2] = pos(y, 0);
        rodPos.set([0, PIVOT_Y, 0, x1, y1, 0, x2, y2, 0]);
        rodGeo.attributes.position.needsUpdate = true;
        bob1.position.set(x1, y1, 0); bob2.position.set(x2, y2, 0);
      },
    };
    return view;
  },

  plots: [
    {
      title: 'Angular positions', x: 't (s)', y: 'angle (°, unwrapped)',
      traces(lab) {
        const A = lab.hist.xy((t) => t, (t, d, o) => d[o] / DEG, { window: 20 });
        const B = lab.hist.xy((t) => t, (t, d, o) => d[o + 1] / DEG, { window: 20 });
        return [line(A.x, A.y, 'θ₁', C.cyan), line(B.x, B.y, 'θ₂', C.amber)];
      },
    },
    {
      title: 'Energy: kinetic, potential, total', x: 't (s)', y: 'energy (J)',
      traces(lab) {
        const p = lab.p, o = { window: 20 };
        const K = lab.hist.xy((t) => t, (t, d, i) => kinetic(d, i, p), o);
        const V = lab.hist.xy((t) => t, (t, d, i) => potential(d, i, p), o);
        const E = lab.hist.xy((t) => t, (t, d, i) => kinetic(d, i, p) + potential(d, i, p), o);
        return [line(K.x, K.y, 'T', C.rose), line(V.x, V.y, 'V', C.cyan), line(E.x, E.y, 'E = T + V', C.green, { line: { color: C.green, width: 1.8 } })];
      },
    },
    {
      title: 'Phase plane (θ₁, ω₁)', x: 'θ₁ (°, wrapped)', y: 'ω₁ (rad/s)',
      traces(lab) {
        const A = lab.hist.xy((t, d, o) => wrap(d[o]) / DEG, (t, d, o) => d[o + 2], { max: 4000 });
        return [dots(A.x, A.y, 'trajectory', C.cyan, 2)];
      },
    },
  ],

  readouts(lab) {
    const y = lab.state(0), p = lab.p, inv = lab.reports[0]?.invariants.energy;
    const E = kinetic(y, 0, p) + potential(y, 0, p);
    return [
      ['θ₁, θ₂ (wrapped)', `${fmt(wrap(y[0]) / DEG, 4)}°, ${fmt(wrap(y[1]) / DEG, 4)}°`],
      ['full turns of θ₁, θ₂', `${Math.round(y[0] / (2 * Math.PI))}, ${Math.round(y[1] / (2 * Math.PI))}`],
      ['ω₁, ω₂ (rad/s)', `${fmt(y[2], 4)}, ${fmt(y[3], 4)}`],
      ['kinetic T (J)', fmt(kinetic(y, 0, p), 5)],
      ['potential V (J)', fmt(potential(y, 0, p), 5)],
      ['total E (J)', fmt(E, 7)],
      ['max |E − E₀| (J)', inv ? fmt(inv.maxAbsDrift, 3) : '—'],
      ['max |E − E₀| / |E₀|', inv && inv.maxRelDrift !== null ? fmt(inv.maxRelDrift, 3) : '—'],
    ];
  },
};

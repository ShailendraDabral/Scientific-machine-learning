import { el, fmt, section } from './dom.js';
import { numberControl, selectControl, button, statusBar } from './controls.js';
import { Plot, baseLayout, line, dots, PLOT_COLORS as C } from './plots.js';
import { EXPLAIN } from './analysisText.js';
import { JobClient } from '../engine/jobs.js';
import { integrators } from '../core/index.js';
import lorenz from '../experiments/lorenz.js';
import rossler from '../experiments/rossler.js';
import duffing from '../experiments/duffing.js';
import doublePendulum from '../experiments/doublePendulum.js';
import threeBody from '../experiments/threeBody.js';

const METHOD = {
  euler: ['Forward Euler', C.rose], midpoint: ['Midpoint (RK2)', '#B39DDB'], rk4: ['Runge–Kutta 4', C.amber],
  symplecticEuler: ['Symplectic Euler', C.cyan], verlet: ['Störmer–Verlet', C.green],
};
const ANALYSES = [
  ['lyapunov', 'Lyapunov exponent and sensitivity'], ['embedding', 'Phase-space reconstruction'],
  ['poincare', 'Poincaré section'], ['compare', 'Integrator comparison'], ['dtStudy', 'Time-step sensitivity'],
];

// Per-system defaults: durations are in the system's own time unit.
const SYS = {
  lorenz: { exp: lorenz, labels: ['x', 'y', 'z'], obs: 0, obs2: 2, published: 0.906,
    lyapunov: { T: 300, transient: 20, renorm: 0.1, sens: 35 }, embedding: { T: 100, transient: 20, sampleDt: 0.02 },
    poincare: { T: 400, transient: 20 }, compare: { T: 3 }, dtStudy: { T: 1, dt: 0.01 } },
  rossler: { exp: rossler, labels: ['x', 'y', 'z'], obs: 0, obs2: 1, published: 0.0714,
    lyapunov: { T: 3000, transient: 200, renorm: 0.5, sens: 300 }, embedding: { T: 600, transient: 100, sampleDt: 0.1 },
    poincare: { T: 3000, transient: 100 }, compare: { T: 20 }, dtStudy: { T: 10, dt: 0.05 } },
  duffing: { exp: duffing, labels: ['x', 'v'], obs: 0, obs2: 1,
    lyapunov: { T: 3000, transient: 100, renorm: 0.25, sens: 150 }, embedding: { T: 500, transient: 100, sampleDt: 0.05 },
    poincare: { T: 6000, transient: 100 }, compare: { T: 20 }, dtStudy: { T: 8, dt: 0.05 } },
  'double-pendulum': { exp: doublePendulum, labels: ['θ₁', 'θ₂', 'ω₁', 'ω₂'], obs: 2, obs2: 3,
    lyapunov: { T: 400, transient: 5, renorm: 0.05, sens: 20 }, embedding: { T: 80, transient: 5, sampleDt: 0.01 },
    poincare: { T: 600, transient: 0 }, compare: { T: 4 }, dtStudy: { T: 1.5, dt: 0.01 } },
  'three-body': { exp: threeBody, labels: ['x₁', 'y₁', 'x₂', 'y₂', 'x₃', 'y₃', 'vx₁', 'vy₁', 'vx₂', 'vy₂', 'vx₃', 'vy₃'], obs: 0, obs2: 6,
    lyapunov: { T: 100, transient: 0, renorm: 0.1, sens: 60 }, embedding: { T: 60, transient: 0, sampleDt: 0.02 },
    poincare: { T: 50, transient: 0 }, compare: { T: 12.65 }, dtStudy: { T: 2, dt: 0.02 } },
};

const logAxis = (title) => ({ type: 'log', exponentformat: 'power', title: { text: title, font: { size: 11 }, standoff: 6 },
  gridcolor: '#1B222C', linecolor: '#2A3340', tickfont: { size: 10 }, automargin: true });
const pos = (v) => (v > 0 ? v : null);
const sci = (v, s = 3) => (v == null ? '—' : fmt(v, s));
const mline = (x, y, m, extra = {}) => ({ ...line(x, y, METHOD[m][0], METHOD[m][1]), ...extra });

/**
 * Analysis laboratory. Every number shown is returned by a job computed in
 * the analysis worker from the numerical core; nothing is tabulated in advance
 * except the two literature values that are labelled as such.
 */
export class AnalysisLab {
  constructor(exp, regions) {
    this.exp = exp; this.R = regions;
    this.jobs = new JobClient();
    this.sysId = 'lorenz'; this.kind = 'lyapunov';
    this.plots = [];
    this.loadSystem();
    this.rebuild();
  }

  get sys() { return SYS[this.sysId]; }

  loadSystem() {
    const e = this.sys.exp;
    this.p = Object.fromEntries(e.params.map((s) => [s.key, s.value]));
    this.ic = Object.fromEntries(e.ic.map((s) => [s.key, s.value]));
    this.method = e.numerics.method;
    this.methods = new Set(e.numerics.methods);
    this.loadAnalysis();
  }

  loadAnalysis() {
    const cfg = this.sys[this.kind];
    this.dt = cfg.dt ?? this.sys.exp.numerics.dt;
    this.T = cfg.T;
    this.lag = 0;
  }

  // ---------------------------------------------------------------- UI

  rebuild() {
    const { R } = this, e = this.sys.exp;
    this.disposePlots();
    for (const k of ['status', 'viewport', 'plots', 'transport', 'inspector']) R[k].replaceChildren();
    R.viewport.classList.add('plot-host');
    R.plots.classList.add('two');
    this.status = statusBar(R.status, [['title', ''], ['state', ''], ['system', 'system'], ['analysis', 'analysis'],
      ['time', 'computed in'], ['engine', 'engine']]);
    this.status.set('title', this.exp.title);
    this.status.set('system', e.title);
    this.status.set('analysis', ANALYSES.find((a) => a[0] === this.kind)[1]);

    this.mainHost = el('div', { class: 'host' });
    this.sideHost = el('div', { class: 'host' });
    this.tableHost = el('div', { class: 'results' });
    this.banner = el('div', { class: 'banner', role: 'status' });
    R.viewport.append(this.mainHost, this.banner);
    R.plots.append(this.sideHost, this.tableHost);

    const stale = () => this.setState('stale');
    const sysSel = selectControl('system', Object.entries(SYS).map(([id, s]) => [id, s.exp.title]), this.sysId,
      (v) => { this.sysId = v; this.loadSystem(); this.rebuild(); });
    const kindSel = selectControl('analysis', ANALYSES, this.kind,
      (v) => { this.kind = v; this.loadAnalysis(); this.rebuild(); });
    const params = e.params.map((s) => numberControl({ ...s, value: this.p[s.key] }, (v) => { this.p[s.key] = v; stale(); }).root);
    const icSl = [], icC = [];
    for (const s of e.ic) {
      if (s.liveOnly) continue; // used only by the live experiment (e.g. the twin-trajectory offset)
      const c = numberControl({ ...s, value: this.ic[s.key] }, (v) => { this.ic[s.key] = v; stale(); });
      (s.slider === false ? icC : icSl).push(c.root);
    }
    const multi = this.kind === 'compare' || this.kind === 'dtStudy';
    const numerics = [];
    if (multi) {
      numerics.push(el('div', { class: 'checks' }, e.numerics.methods.map((m) => {
        const box = el('input', { type: 'checkbox', checked: this.methods.has(m) });
        box.addEventListener('change', () => { if (box.checked) this.methods.add(m); else this.methods.delete(m); stale(); });
        return el('label', { class: 'check' }, box, el('i', { style: `background:${METHOD[m][1]}` }), METHOD[m][0]);
      })));
    } else {
      numerics.push(selectControl('method', e.numerics.methods.map((m) => [m, METHOD[m][0]]), this.method,
        (m) => { this.method = m; stale(); }).root);
    }
    numerics.push(
      numberControl({ key: 'dt', label: this.kind === 'dtStudy' ? 'largest d<i>t</i>' : 'd<i>t</i>', value: this.dt,
        min: e.numerics.dtMin, max: e.numerics.dtMax, log: true, aria: 'time step' }, (v) => { this.dt = v; stale(); }).root,
      numberControl({ key: 'T', label: '<i>T</i>', unit: 'duration', value: this.T, slider: false, min: 0, max: 1e6, aria: 'duration' },
        (v) => { this.T = v; stale(); }).root);
    if (this.kind === 'embedding') {
      numerics.push(numberControl({ key: 'lag', label: 'lag', unit: 'samples', value: this.lag, slider: false, min: 0, max: 1e4, aria: 'delay' },
        (v) => { this.lag = Math.max(0, Math.round(v)); stale(); }).root,
      el('p', { class: 'note' }, `Delay in samples of ${this.sys.embedding.sampleDt}; 0 selects it from the autocorrelation. Observable: ${this.sys.labels[this.sys.obs]}.`));
    }
    R.inspector.append(
      section('Analysis', sysSel.root, kindSel.root),
      section('Parameters', ...params),
      section('Initial conditions', ...icSl, icC.length ? el('div', { class: 'ctl-grid' }, ...icC) : null),
      section(multi ? 'Methods and numerics' : 'Numerics', ...numerics),
      section('What this measures', el('div', { class: 'explain', html: EXPLAIN[this.kind] })),
    );

    this.runBtn = button('Run analysis', 'Compute with the current settings (Enter)', () => this.run(), 'primary');
    this.note = el('span', { class: 'transport-note' });
    R.transport.append(this.runBtn, this.note);
    this.onKey = (ev) => { if (ev.key === 'Enter' && !ev.target.matches('select')) this.run(); };
    window.removeEventListener('keydown', this.boundKey);
    this.boundKey = this.onKey;
    window.addEventListener('keydown', this.boundKey);
    this.run();
  }

  setState(state, text = '') {
    const map = { running: ['COMPUTING', 'warn'], done: ['DONE', 'good'], stale: ['SETTINGS CHANGED', 'idle'], error: ['ERROR', 'bad'] };
    this.status.set('state', map[state][0], map[state][1]);
    this.status.set('engine', this.jobs.mode);
    if (state === 'stale') this.note.textContent = 'settings changed — press Run analysis to recompute';
    else if (state === 'running') this.note.textContent = 'computing…';
    else this.note.textContent = text;
    this.banner.textContent = state === 'error' ? text : '';
    this.banner.className = `banner${state === 'error' ? ' show' : ''}`;
  }

  // ---------------------------------------------------------------- job

  job() {
    const s = this.sys, e = s.exp, cfg = s[this.kind];
    const params = e.coreParams(this.p);
    const y0 = Array.from(e.initialStates(this.p, this.ic)[0]);
    const base = { kind: this.kind, systemId: e.system, params, y0, dt: this.dt, method: this.method, duration: this.T,
      transient: cfg.transient ?? 0 };
    if (this.kind === 'lyapunov') return { ...base, renormTime: cfg.renorm, divergenceDuration: cfg.sens, deltas: [1e-10, 1e-7, 1e-4] };
    if (this.kind === 'embedding') return { ...base, sampleDt: cfg.sampleDt, index: s.obs, index2: s.obs2, lag: this.lag || null };
    if (this.kind === 'compare' || this.kind === 'dtStudy') {
      return { ...base, methods: e.numerics.methods.filter((m) => this.methods.has(m)) };
    }
    return base;
  }

  async run() {
    let job;
    try {
      job = this.job();
      if (job.methods && !job.methods.length) throw new Error('select at least one method');
      if (!(job.duration > 0)) throw new Error('duration must be a number > 0');
    } catch (err) { return this.setState('error', `${err.name}: ${err.message}`); }
    this.setState('running');
    const gen = (this.gen = (this.gen || 0) + 1);
    const res = await this.jobs.run(job);
    if (gen !== this.gen || res.stale) return;
    this.status.set('time', `${fmt(res.ms, 3)} ms`);
    if (res.error) return this.setState('error', `${res.error.name}: ${res.error.message}`);
    this.result = res.result;
    this.render(job, res.result);
    this.setState('done', `computed from ${this.describe(job)}`);
  }

  describe(job) {
    return `${this.sys.exp.title}, dt = ${fmt(job.dt, 3)}, duration ${fmt(job.duration, 4)}`
      + (job.methods ? '' : `, ${METHOD[job.method][0]}`);
  }

  // ---------------------------------------------------------------- rendering

  disposePlots() { for (const p of this.plots) p.dispose(); this.plots = []; }

  plot(host, layout, traces) {
    const div = el('div', { class: 'plot fill' });
    host.replaceChildren(div);
    const p = new Plot(div, layout);
    p.update(traces);
    this.plots.push(p);
  }

  message(host, text) { host.replaceChildren(el('div', { class: 'message' }, text)); }

  table(title, head, rows, notes = []) {
    this.tableHost.replaceChildren(
      el('h3', {}, title),
      el('table', {},
        head ? el('thead', {}, el('tr', {}, head.map((h) => el('th', { html: h })))) : null,
        el('tbody', {}, rows.map((r) => el('tr', {}, r.map((c, i) => el(i === 0 ? 'th' : 'td', { html: String(c) })))))),
      ...notes.map((n) => el('p', { class: 'note', html: n })));
  }

  render(job, r) {
    this.disposePlots();
    this[`render_${this.kind}`](job, r);
  }

  render_lyapunov(job, { lyapunov: L, divergence: D }) {
    const colors = [C.cyan, C.amber, C.rose];
    const traces = D.series.map((s, i) => line(D.t, s.log10d, `δ₀ = ${fmt(s.delta, 1)}`, colors[i]));
    let top = -Infinity;
    for (const s of D.series) for (const v of s.log10d) if (v !== null && v > top) top = v;
    if (L.lambda > 0) {
      const d0 = Math.log10(D.series[1].delta), tEnd = Math.min(D.t.at(-1), ((top - d0) * Math.LN10) / L.lambda);
      traces.push(line([0, tEnd], [d0, d0 + (L.lambda * tEnd) / Math.LN10], `slope λ₁/ln 10, λ₁ = ${fmt(L.lambda, 3)}`, C.green,
        { line: { color: C.green, width: 1, dash: 'dash' } }));
    }
    this.plot(this.mainHost, baseLayout({ title: 'Sensitivity to initial conditions: separation of neighbouring trajectories',
      x: 't after the transient', y: 'log₁₀ |δ(t)|' }), traces);
    const tt = L.series.t;
    this.plot(this.sideHost, baseLayout({ title: 'Running estimate of λ₁', x: 'averaging time T', y: 'λ₁(T)' }), [
      line(tt, L.series.lambda, 'λ₁(T)', C.amber),
      line([tt[0], tt.at(-1)], [L.lambda + L.stderr, L.lambda + L.stderr], '± standard error', C.grey, { line: { color: C.grey, width: 1, dash: 'dot' } }),
      line([tt[0], tt.at(-1)], [L.lambda - L.stderr, L.lambda - L.stderr], '', C.grey, { line: { color: C.grey, width: 1, dash: 'dot' }, showlegend: false }),
    ]);
    const z = L.stderr > 0 ? L.lambda / L.stderr : 0;
    const verdict = z > 3 ? ['positive: chaotic', 'bad']
      : z < -3 ? ['negative: trajectories converge', 'good']
        : ['not distinguishable from zero in this run', 'warn'];
    const rows = [
      ['λ₁ (largest exponent)', `${fmt(L.lambda, 4)} ± ${fmt(L.stderr, 2)}`],
      ['verdict', `<span class="${verdict[1]}">${verdict[0]}</span>`],
    ];
    if (z > 3) {
      const scale = 10 ** top;
      rows.push(['Lyapunov time 1/λ₁', fmt(1 / L.lambda, 4)], ['error doubling time ln 2/λ₁', fmt(Math.LN2 / L.lambda, 4)],
        ['largest separation observed', fmt(scale, 3)],
        ['predictability horizon for δ₀ = 10⁻¹⁶', `${fmt(Math.log(scale / 1e-16) / L.lambda, 4)}`]);
    }
    rows.push(['averaging time', fmt(L.duration, 5)], ['renormalisation interval τ', fmt(L.renormTime, 3)],
      ['intervals', String(L.intervals)], ['initial separation d₀', fmt(L.d0, 1)], ['transient discarded', fmt(L.transient, 4)],
      ['max dt·rate / limit', fmt(L.stability.maxDtRate / L.stability.limit, 3)]);
    const notes = [];
    const e = this.sys.exp, atDefaults = e.params.every((s) => this.p[s.key] === s.value);
    if (this.sys.published && atDefaults) {
      rows.push(['published value at these parameters', String(this.sys.published)]);
      notes.push('The published value is quoted from the literature for comparison; it is not computed here.');
    }
    if (z > 3) notes.push('The horizon is ln(largest separation / δ₀) / λ₁: the time for a round-off-sized error to reach the size of the attractor.');
    this.table('Lyapunov exponent', null, rows, notes);
  }

  render_embedding(job, r) {
    const lab = this.sys.labels[this.sys.obs], tau = r.lag * r.sampleDt;
    const scene = { bgcolor: 'rgba(0,0,0,0)' };
    for (const [k, t] of [['xaxis', `${lab}(t)`], ['yaxis', `${lab}(t+τ)`], ['zaxis', `${lab}(t+2τ)`]]) {
      scene[k] = { title: { text: t, font: { size: 11 } }, gridcolor: '#232A35', zerolinecolor: '#2F3947', showbackground: false, tickfont: { size: 9 } };
    }
    this.plot(this.mainHost, baseLayout({ title: `Delay reconstruction from ${lab}(t) alone, τ = ${fmt(tau, 3)}`, x: '', y: '',
      extra: { scene, margin: { l: 0, r: 0, t: 40, b: 0 }, showlegend: false } }), [
      { type: 'scatter3d', mode: 'lines', x: r.cols[0], y: r.cols[1], z: r.cols[2], line: { color: C.amber, width: 1.5 }, hoverinfo: 'skip' },
    ]);
    const lags = r.acf.map((_, k) => k * r.sampleDt);
    const lo = Math.min(...r.acf);
    this.plot(this.sideHost, baseLayout({ title: `Autocorrelation of ${lab}(t)`, x: 'lag', y: 'C(lag)' }), [
      line(lags, r.acf, 'C', C.amber),
      line([lags[0], lags.at(-1)], [1 / Math.E, 1 / Math.E], '1/e', C.grey, { line: { color: C.grey, width: 1, dash: 'dot' } }),
      line([tau, tau], [Math.min(lo, 0), 1], `chosen τ = ${fmt(tau, 3)}`, C.cyan, { line: { color: C.cyan, width: 1, dash: 'dash' } }),
    ]);
    this.table('Reconstruction', null, [
      ['observable', lab], ['samples', String(r.samples)], ['sampling interval', fmt(r.sampleDt, 3)],
      ['delay τ', `${fmt(tau, 4)} (${r.lag} samples)`], ['delay rule', r.rule],
      ['automatic delay', `${fmt(r.autoLag * r.sampleDt, 4)} (${r.autoLag} samples)`],
      ['embedding dimension m', '3 (fixed)'], ['C at chosen delay', fmt(r.acf[Math.min(r.lag, r.acf.length - 1)], 3)],
    ], ['Rotate the 3D plot: the reconstructed attractor should show the same number of lobes, holes and folds as the true one, in distorted form.']);
  }

  render_poincare(job, r) {
    if (!r.applicable) {
      this.message(this.mainHost, `Poincaré section not applicable: ${r.reason}.`);
      this.message(this.sideHost, 'No return map.');
      return this.table('Poincaré section', null, [['status', 'not applicable'], ['reason', r.reason]]);
    }
    this.plot(this.mainHost, baseLayout({ title: `Poincaré section: ${r.label}`, x: r.axes[0], y: r.axes[1], extra: { showlegend: false } }),
      [dots(r.a, r.b, 'crossings', C.cyan, r.points > 1500 ? 2.5 : 4)]);
    this.plot(this.sideHost, baseLayout({ title: `Return map of ${r.axes[0]}`, x: `${r.axes[0]} at crossing n`, y: 'at crossing n + 1',
      extra: { showlegend: false } }), [dots(r.a.slice(0, -1), r.a.slice(1), 'return map', C.amber, 3)]);
    let distinct = 0;
    if (r.points) {
      const seen = new Set();
      const span = (Math.max(...r.a) - Math.min(...r.a)) || 1;
      for (let i = Math.floor(r.points / 2); i < r.points; i++) seen.add(`${Math.round((r.a[i] / span) * 2000)}:${Math.round((r.b[i] / span) * 2000)}`);
      distinct = seen.size;
    }
    this.table('Poincaré section', null, [
      ['section', r.label], ['points recorded', String(r.points)],
      ['distinct points in the second half', `${distinct} (to 1 part in 2000 of the range)`],
      ['trajectory length', fmt(r.duration, 5)], ['transient discarded', fmt(r.transient, 4)],
      ['crossing residual max |g|', r.kind === 'strobe' ? 'exact time (strobe)' : sci(r.residual, 2)],
      ['interpolation', 'cubic Hermite within the step'],
    ], ['A small number of distinct points indicates a periodic orbit of that period; a number close to the points recorded indicates quasi-periodic or chaotic motion.']);
  }

  render_compare(job, r) {
    const ok = r.results.filter((m) => m.applicable);
    this.plot(this.mainHost, baseLayout({ title: `Trajectory difference from the reference (RK4, dt = ${fmt(r.reference.dt, 3)})`,
      x: 't', y: '', extra: { yaxis: logAxis('‖y − y_ref‖') } }),
    [...ok.map((m) => mline(r.t.slice(0, m.err.length), m.err.map(pos), m.method)),
      ...(r.reference.selfDifference > 0 ? [line([r.t[0], r.t.at(-1)], [r.reference.selfDifference, r.reference.selfDifference],
        'resolution of the reference', C.grey, { line: { color: C.grey, width: 1, dash: 'dot' } })] : [])]);
    if (r.hasEnergy) {
      this.plot(this.sideHost, baseLayout({ title: `Energy error ${r.energyRelative ? '|E − E₀| / |E₀|' : '|E − E₀| (E₀ ≈ 0)'}`, x: 't', y: '',
        extra: { yaxis: logAxis('energy error') } }),
      ok.map((m) => mline(r.t.slice(0, m.energyErr.length), m.energyErr.map(pos), m.method)));
    } else {
      this.message(this.sideHost, 'Energy conservation does not apply: with these parameters the model is dissipative or driven, so it has no conserved energy to check.');
    }
    const rows = r.results.map((m) => {
      const name = `<i class="sw" style="background:${METHOD[m.method][1]}"></i>${METHOD[m.method][0]}`;
      if (!m.applicable) return [name, `<span class="warn">not applicable</span>`, '—', '—', '—', '—', '—'];
      const st = m.status === 'stable' ? '<span class="good">stable</span>'
        : m.status === 'diverged' ? `<span class="bad">diverged at t = ${fmt(m.failure.t, 3)}</span>` : '<span class="warn">above linear limit</span>';
      return [name, st, fmt(m.maxDtRate / m.limit, 2), sci(m.finalErr, 2), sci(m.maxErr, 2),
        r.hasEnergy ? sci(m.maxEnergyErr, 2) : 'n/a', m.nsPerStep === null ? '—' : `${fmt(m.nsPerStep, 3)} ns · ${fmt(m.runMs, 3)} ms`];
    });
    const notes = [`Reference self-difference (RK4 at dt/16 against dt/32): ${sci(r.reference.selfDifference, 2)}. ${r.steps} steps of dt = ${fmt(r.dt, 3)} per method.`];
    for (const m of r.results) if (!m.applicable) notes.push(`${METHOD[m.method][0]}: ${m.reason}.`);
    this.table('Integrator comparison', ['method', 'stability', 'dt·rate / limit', 'final diff.', 'max diff.', 'max energy err.', 'time per step · run'], rows, notes);
  }

  render_dtStudy(job, r) {
    const ok = r.results.filter((m) => m.applicable);
    this.plot(this.mainHost, baseLayout({ title: `Error at t = ${fmt(r.duration, 4)} against step size`, x: '', y: '',
      extra: { xaxis: logAxis('dt'), yaxis: logAxis('‖y − y_ref‖ at the end time') } }),
    [...ok.map((m) => mline(r.dts, m.errors.map(pos), m.method, { mode: 'lines+markers', marker: { size: 5, color: METHOD[m.method][1] },
      name: `${METHOD[m.method][0]}${m.order === null ? '' : ` · slope ${m.order.toFixed(2)}`}` })),
    line([r.dts[0], r.dts.at(-1)], [r.floor, r.floor], 'fit threshold (10 × reference resolution)', C.grey, { line: { color: C.grey, width: 1, dash: 'dot' } })]);
    this.plot(this.sideHost, baseLayout({ title: 'Work–precision: error against computing time', x: '', y: '',
      extra: { xaxis: logAxis('computing time (ms, measured)'), yaxis: logAxis('error') } }),
    ok.filter((m) => m.nsPerStep !== null).map((m) => mline(m.costMs, m.errors.map(pos), m.method,
      { mode: 'lines+markers', marker: { size: 5, color: METHOD[m.method][1] } })));
    const rows = r.results.map((m) => {
      const name = `<i class="sw" style="background:${METHOD[m.method][1]}"></i>${METHOD[m.method][0]}`;
      if (!m.applicable) return [name, '—', '<span class="warn">not applicable</span>', '—', '—', '—'];
      const failed = m.failures.filter(Boolean).length;
      const cls = m.order === null ? 'warn' : Math.abs(m.order - m.nominalOrder) < 0.3 ? 'good' : 'warn';
      return [name, String(m.nominalOrder),
        `<span class="${cls}">${m.order === null ? 'no fit' : m.order.toFixed(2)}</span> (${m.fitPoints} pts)`,
        sci(m.errors[0], 2), sci(m.errors.at(-1), 2),
        (r.hasEnergy ? sci(m.energies.at(-1), 2) : 'n/a') + (failed ? ` <span class="bad">${failed} diverged</span>` : '')];
    });
    const notes = [`Step sizes ${fmt(r.dts[0], 3)} down to ${fmt(r.dts.at(-1), 3)} (halved ${r.dts.length - 1} times). Reference: RK4 at dt = ${fmt(r.reference.dt, 3)}, self-difference ${sci(r.reference.selfDifference, 2)}.`];
    for (const m of r.results) if (!m.applicable) notes.push(`${METHOD[m.method][0]}: ${m.reason}.`);
    this.table('Observed order of accuracy', ['method', 'nominal order', 'observed slope', 'error, largest dt', 'error, smallest dt', 'energy err., smallest dt'], rows, notes);
  }

  destroy() {
    this.gen = (this.gen || 0) + 1;
    this.jobs.dispose();
    window.removeEventListener('keydown', this.boundKey);
    this.disposePlots();
    this.R.viewport.classList.remove('plot-host');
    this.R.plots.classList.remove('two');
  }
}

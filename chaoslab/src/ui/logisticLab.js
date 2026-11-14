import { el, fmt, tex, section } from './dom.js';
import { numberControl, button, readouts, statusBar } from './controls.js';
import { Plot, baseLayout, line, dots, PLOT_COLORS as C } from './plots.js';
import { logistic } from '../core/index.js';

const R_DEFAULT = [2.5, 4];
const TRANSIENT = 600, KEEP = 300;

/** Classify the long-run behaviour at r from an actual orbit. */
function classify(r) {
  const lambda = logistic.lyapunov({ params: { r }, x0: 0.3141, n: 20000, discard: 5000 });
  const o = logistic.orbit({ params: { r }, x0: 0.3141, n: 1024, discard: 20000 });
  let period = 0;
  for (let k = 1; k <= 256 && !period; k++) {
    let ok = true;
    for (let i = 0; i < 512 && ok; i++) ok = Math.abs(o[i + k] - o[i]) < 1e-7;
    if (ok) period = k;
  }
  let label, cls = 'good';
  if (period === 1) label = o[0] < 1e-6 ? 'stable: x → 0 (extinction)' : `stable fixed point x* = ${fmt(o[0], 6)}`;
  else if (period > 1) label = `stable cycle, period ${period}`;
  else if (lambda > 0) { label = 'chaotic (λ > 0)'; cls = 'bad'; }
  else { label = 'periodic: long period or slow convergence'; cls = 'warn'; }
  return { lambda, period, label, cls };
}

/**
 * Laboratory for the logistic map. The orbit is iterated with the core map
 * function; the bifurcation diagram and Lyapunov curve are computed from
 * orbits at each r. Nothing is precomputed or drawn from a formula.
 */
export class LogisticLab {
  constructor(exp, regions) {
    this.exp = exp; this.R = regions;
    this.r = 3.7; this.x0 = 0.2; this.shown = 60;
    this.range = [...R_DEFAULT]; this.yRange = [0, 1];
    this.speed = 1; this.running = true;
    this.acc = 0; this.last = performance.now(); this.lastPlot = 0; this.dirty = true;
    this.error = null;
    this.build();
    this.computeDiagram();
    this.reset();
    this.frame = this.frame.bind(this);
    this.raf = requestAnimationFrame(this.frame);
  }

  build() {
    const { R, exp } = this;
    R.viewport.classList.add('plot-host');
    R.plots.classList.add('two');
    this.status = statusBar(R.status, [['title', ''], ['state', ''], ['n', 'n'], ['x', 'xₙ'], ['r', 'r'], ['regime', 'regime'], ['rate', 'rate']]);
    this.status.set('title', exp.title);

    const bifDiv = el('div', { class: 'plot fill' });
    R.viewport.append(bifDiv);
    const ax = { gridcolor: '#1B222C', zerolinecolor: '#3A4656', linecolor: '#2A3340', tickfont: { size: 10 }, automargin: true };
    this.bif = new Plot(bifDiv, baseLayout({
      title: 'Bifurcation diagram and Lyapunov exponent', x: 'r', y: 'x on the attractor',
      extra: {
        yaxis: { ...ax, domain: [0.3, 1], range: [0, 1], title: { text: 'x on the attractor', font: { size: 11 } } },
        yaxis2: { ...ax, domain: [0, 0.22], title: { text: 'λ(r)', font: { size: 11 } } },
        margin: { l: 52, r: 14, t: 46, b: 38 },
      },
    }));
    this.bif.on('plotly_relayout', (ev) => this.onZoom(ev));

    const cobDiv = el('div', { class: 'plot' }), tsDiv = el('div', { class: 'plot' });
    R.plots.append(cobDiv, tsDiv);
    this.cob = new Plot(cobDiv, baseLayout({ title: 'Cobweb diagram', x: 'xₙ', y: 'xₙ₊₁',
      extra: { xaxis: { range: [0, 1], gridcolor: '#1B222C', linecolor: '#2A3340', title: { text: 'xₙ', font: { size: 11 } }, tickfont: { size: 10 } },
        yaxis: { range: [0, 1], gridcolor: '#1B222C', linecolor: '#2A3340', title: { text: 'xₙ₊₁', font: { size: 11 } }, tickfont: { size: 10 } } } }));
    this.ts = new Plot(tsDiv, baseLayout({ title: 'Orbit xₙ', x: 'n', y: 'xₙ' }));

    this.rCtl = numberControl({ key: 'r', tex: 'r', value: this.r, min: 0, max: 4, step: 0.001 }, (v) => this.setR(v));
    this.x0Ctl = numberControl({ key: 'x0', tex: 'x_0', value: this.x0, min: 0, max: 1, step: 0.001 },
      (v) => { this.x0 = v; this.reset(); });
    const shownCtl = numberControl({ key: 'shown', label: 'iterates shown', value: this.shown, min: 5, max: 400, step: 1 },
      (v) => { this.shown = Math.max(1, Math.round(v)); this.dirty = true; });
    this.rMin = numberControl({ key: 'rmin', tex: 'r_{\\min}', value: this.range[0], slider: false, min: 0, max: 4 },
      (v) => this.setRange(v, this.range[1]));
    this.rMax = numberControl({ key: 'rmax', tex: 'r_{\\max}', value: this.range[1], slider: false, min: 0, max: 4 },
      (v) => this.setRange(this.range[0], v));
    this.readouts = readouts();
    this.banner = el('div', { class: 'banner', role: 'status' });
    R.viewport.append(this.banner);

    const presets = [['3.7', 'r = 3.7 · chaotic'], ['2.8', 'r = 2.8 · stable fixed point'], ['3.2', 'r = 3.2 · period 2'],
      ['3.5', 'r = 3.5 · period 4'], ['3.5699', 'r = 3.5699 · onset of chaos'], ['3.835', 'r = 3.835 · period-3 window'], ['4', 'r = 4 · fully chaotic']];
    const sel = el('select', { class: 'ctl-select', 'aria-label': 'preset' }, presets.map(([v, t]) => el('option', { value: v }, t)));
    sel.addEventListener('change', () => { this.rCtl.set(Number(sel.value)); this.setR(Number(sel.value)); this.reset(); });

    R.inspector.append(
      section('Presets', el('div', { class: 'ctl ctl-row' }, el('label', { class: 'ctl-label' }, 'preset'), sel)),
      section('Parameter', this.rCtl.root),
      section('Initial condition', this.x0Ctl.root, shownCtl.root),
      section('Bifurcation range', el('div', { class: 'ctl-grid' }, this.rMin.root, this.rMax.root),
        button('Full range 2.5 – 4', 'Reset the diagram range', () => this.setRange(...R_DEFAULT)),
        el('p', { class: 'note' }, `Each column: ${TRANSIENT} iterates discarded, ${KEEP} kept, started from x = 0.5.`)),
      section('Readouts', this.readouts.root),
      section('Equations', el('div', { class: 'equations', html: exp.tex.map((t) => tex(t, true)).join('') }),
        el('p', { class: 'note' }, exp.blurb)),
    );

    this.playBtn = button('Pause', 'Pause or resume (Space)', () => this.toggle(), 'primary');
    this.speedLabel = el('span', { class: 'speed-value' }, '8 /s');
    const speed = el('input', { type: 'range', min: -3, max: 5, step: 0.25, value: 0, class: 'speed', 'aria-label': 'iteration rate' });
    speed.addEventListener('input', () => {
      this.speed = 2 ** parseFloat(speed.value);
      this.speedLabel.textContent = `${fmt(8 * this.speed, 3)} /s`;
    });
    R.transport.append(
      button('Reset', 'Restart from x₀ (R)', () => this.reset()),
      button('Step', 'One iteration (.)', () => { this.running = false; this.iterate(1); this.sync(); }),
      this.playBtn,
      el('label', { class: 'speed-wrap' }, 'iterations', speed, this.speedLabel),
      el('span', { class: 'transport-note' }, 'discrete map: no time step; each iteration is exact up to round-off'));

    this.onKey = (e) => {
      if (e.target.matches('input, select, textarea')) return;
      if (e.code === 'Space') { e.preventDefault(); this.toggle(); }
      else if (e.key === 'r' || e.key === 'R') this.reset();
      else if (e.key === '.') { this.running = false; this.iterate(1); this.sync(); }
    };
    window.addEventListener('keydown', this.onKey);
  }

  // ---------------------------------------------------------------- model

  validate() {
    try {
      logistic.orbit({ params: { r: this.r }, x0: this.x0, n: 0 });
      this.error = null;
    } catch (e) { this.error = `${e.name}: ${e.message}`; }
    return !this.error;
  }

  setR(v) {
    this.r = v;
    if (this.validate()) {
      this.regime = classify(this.r);
      this.bif.relayout({ shapes: this.marker() });
    }
    this.sync();
  }

  reset() {
    if (!this.validate()) return this.sync();
    this.orbit = [this.x0];
    this.n = 0; this.acc = 0;
    this.regime = classify(this.r);
    this.sync();
  }

  /** Append k iterates using the core map; the parameter may change between calls. */
  iterate(k) {
    if (this.error) return;
    const p = { r: this.r };
    let x = this.orbit[this.orbit.length - 1];
    for (let i = 0; i < k; i++) { x = logistic.map(x, p); this.orbit.push(x); this.n++; }
    if (this.orbit.length > 4000) this.orbit.splice(0, this.orbit.length - 2000);
    this.dirty = true;
  }

  // ---------------------------------------------------------------- diagram

  setRange(a, b) {
    if (!(Number.isFinite(a) && Number.isFinite(b)) || a < 0 || b > 4 || b - a < 1e-6) {
      this.error = 'InvalidParameterError: the diagram range needs 0 ≤ r_min < r_max ≤ 4';
      return this.sync();
    }
    this.error = null;
    this.range = [a, b];
    this.rMin.set(a); this.rMax.set(b);
    this.computeDiagram();
    this.sync();
  }

  onZoom(ev) {
    if (this.internal) return;
    if (ev['xaxis.autorange']) { this.yRange = [0, 1]; return this.setRange(...R_DEFAULT); }
    const a = ev['xaxis.range[0]'], b = ev['xaxis.range[1]'];
    if (a === undefined || b === undefined) return;
    if (ev['yaxis.range[0]'] !== undefined) {
      this.yRange = [Math.max(0, ev['yaxis.range[0]']), Math.min(1, ev['yaxis.range[1]'])];
    }
    this.setRange(Math.max(0, Math.min(a, b)), Math.min(4, Math.max(a, b)));
  }

  marker() {
    return [{ type: 'line', xref: 'x', yref: 'paper', x0: this.r, x1: this.r, y0: 0, y1: 1,
      line: { color: C.amber, width: 1 } }];
  }

  computeDiagram() {
    const [a, b] = this.range;
    const cols = Math.max(300, Math.min(1400, Math.round(this.R.viewport.clientWidth || 900)));
    const rs = [], xs = [], lr = [], lam = [];
    const seen = new Set();
    for (let c = 0; c < cols; c++) {
      const r = a + ((b - a) * c) / (cols - 1), p = { r };
      let x = 0.5;
      for (let i = 0; i < TRANSIENT; i++) x = logistic.map(x, p);
      let sum = 0;
      seen.clear();
      for (let i = 0; i < KEEP; i++) {
        const key = Math.round(x * 4e4); // merge points closer than a pixel
        if (!seen.has(key)) { seen.add(key); rs.push(r); xs.push(x); }
        sum += Math.log(Math.abs(logistic.derivative(x, p)));
        x = logistic.map(x, p);
      }
      lr.push(r); lam.push(Math.max(-3, sum / KEEP));
    }
    this.points = rs.length;
    this.internal = true;
    this.bif.update([
      { x: rs, y: xs, type: 'scattergl', mode: 'markers', name: 'iterates', hoverinfo: 'skip',
        marker: { color: '#C9D1DB', size: 1.6, opacity: 0.55 } },
      { x: lr, y: lam.map((v) => Math.min(v, 0)), yaxis: 'y2', type: 'scatter', mode: 'lines', name: 'λ < 0 · stable / periodic',
        fill: 'tozeroy', fillcolor: 'rgba(76,201,240,0.28)', line: { color: C.cyan, width: 1 }, hoverinfo: 'skip' },
      { x: lr, y: lam.map((v) => Math.max(v, 0)), yaxis: 'y2', type: 'scatter', mode: 'lines', name: 'λ > 0 · chaotic',
        fill: 'tozeroy', fillcolor: 'rgba(240,98,146,0.35)', line: { color: C.rose, width: 1 }, hoverinfo: 'skip' },
    ], {
      uirevision: `${a}:${b}:${this.yRange}`,
      xaxis: { ...this.bif.layout.xaxis, range: [a, b], autorange: false },
      yaxis: { ...this.bif.layout.yaxis, range: [...this.yRange], autorange: false },
      shapes: this.marker(),
    });
    this.internal = false;
  }

  // ---------------------------------------------------------------- loop

  toggle() { this.running = !this.running; this.sync(); }

  frame(now) {
    this.raf = requestAnimationFrame(this.frame);
    const wall = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    if (this.running && !this.error) {
      this.acc += wall * 8 * this.speed;
      const k = Math.floor(this.acc);
      if (k > 0) { this.acc -= k; this.iterate(k); }
    }
    if (this.dirty && now - this.lastPlot > 80) { this.lastPlot = now; this.draw(); }
  }

  sync() { this.dirty = true; this.draw(); }

  draw() {
    this.dirty = false;
    this.playBtn.textContent = this.running ? 'Pause' : 'Resume';
    const s = this.status;
    s.set('state', this.error ? 'ERROR' : this.running ? 'RUNNING' : 'PAUSED', this.error ? 'bad' : this.running ? 'good' : 'idle');
    this.banner.textContent = this.error || '';
    this.banner.className = `banner${this.error ? ' show' : ''}`;
    if (this.error || !this.orbit) return;
    const o = this.orbit, N = o.length, x = o[N - 1], r = this.r;
    s.set('n', String(this.n)); s.set('x', fmt(x, 6)); s.set('r', fmt(r, 6));
    s.set('regime', this.regime.label, this.regime.cls);
    s.set('rate', `${fmt(8 * this.speed, 3)} iterations/s`);

    // cobweb: the last `shown` iterates of the simulated orbit
    const start = Math.max(0, N - 1 - this.shown), cx = [], cy = [];
    cx.push(o[start]); cy.push(start === 0 ? 0 : o[start]);
    for (let i = start; i < N - 1; i++) { cx.push(o[i], o[i + 1]); cy.push(o[i + 1], o[i + 1]); }
    const fx = [], fy = [], p = { r };
    for (let i = 0; i <= 200; i++) { fx.push(i / 200); fy.push(logistic.map(i / 200, p)); }
    this.cob.update([
      line(fx, fy, 'f(x) = r x (1 − x)', C.cyan, { line: { color: C.cyan, width: 1.4 } }),
      line([0, 1], [0, 1], 'y = x', C.grey),
      line(cx, cy, 'orbit', C.amber, { line: { color: C.amber, width: 0.9 } }),
      dots([x], [x], 'current xₙ', C.amber, 7, { showlegend: false }),
    ]);
    const t0 = Math.max(0, N - 1 - Math.max(this.shown, 120)), nx = [], ny = [];
    for (let i = t0; i < N; i++) { nx.push(this.n - (N - 1 - i)); ny.push(o[i]); }
    this.ts.update([{ ...line(nx, ny, 'xₙ', C.amber), mode: 'lines+markers', marker: { size: 3, color: C.amber } }]);

    const fp = logistic.fixedPoints(p);
    this.readouts.update([
      ['iteration n', String(this.n)],
      ['xₙ', fmt(x, 10)],
      ['regime at this r', this.regime.label, this.regime.cls],
      ['Lyapunov exponent λ(r)', fmt(this.regime.lambda, 5)],
      ['fixed point x* = 1 − 1/r', fp.length > 1 ? fmt(fp[1], 8) : 'none (r ≤ 1)'],
      ['multiplier |f′(x*)| = |2 − r|', fp.length > 1 ? `${fmt(Math.abs(2 - r), 5)} (${Math.abs(2 - r) < 1 ? 'stable' : 'unstable'})` : '—'],
      ['diagram points', String(this.points)],
    ]);
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('keydown', this.onKey);
    this.bif.dispose(); this.cob.dispose(); this.ts.dispose();
    this.R.viewport.classList.remove('plot-host');
    this.R.plots.classList.remove('two');
  }
}

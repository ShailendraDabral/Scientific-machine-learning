import { SceneManager } from '../render/scene.js';
import { el, fmt, tex, section } from './dom.js';
import { numberControl, selectControl, button, readouts, statusBar } from './controls.js';
import { Plot, baseLayout } from './plots.js';
import { History } from './history.js';
import { integrators } from '../core/index.js';

const METHOD_NAMES = {
  euler: 'Forward Euler', midpoint: 'Midpoint (RK2)', rk4: 'Runge–Kutta 4',
  symplecticEuler: 'Symplectic Euler', verlet: 'Störmer–Verlet',
};
const MAX_STEPS_PER_REQUEST = 200000;
const SAMPLES_PER_REQUEST = 300;

/**
 * Generic laboratory for continuous-time systems. It owns the controls, the
 * frame loop and the sample history; the experiment module supplies the model
 * wiring, the 3D view and the plot definitions. All numbers shown come from
 * the engine's replies.
 */
export class OdeLab {
  constructor(exp, regions, client) {
    this.exp = exp; this.R = regions; this.client = client;
    this.p = Object.fromEntries(exp.params.map((s) => [s.key, s.value]));
    this.ic = Object.fromEntries(exp.ic.map((s) => [s.key, s.value]));
    this.dt = exp.numerics.dt;
    this.method = exp.numerics.method;
    this.speed = 1;
    this.running = true;
    this.error = null;
    this.ready = false;
    this.pending = false;
    this.gen = 0;
    this.acc = 0;
    this.t = 0; this.steps = 0;
    this.states = null; this.reports = [];
    this.achieved = 0; this.rateT = 0; this.rateWall = 0;
    this.section = [];
    this.store = {}; // scratch space for the experiment's derived series
    this.dirty = true;
    this.last = performance.now(); this.lastPlot = 0;
    this.build();
    this.reset();
    this.frame = this.frame.bind(this);
    this.raf = requestAnimationFrame(this.frame);
  }

  // ---------------------------------------------------------------- UI

  build() {
    const { exp, R } = this;
    this.status = statusBar(R.status, [
      ['title', ''], ['state', ''], ['t', 't'], ['steps', 'steps'], ['dt', 'dt'], ['method', 'method'],
      ['speed', 'speed'], ['stab', 'dt·rate / limit'], ['engine', 'engine'],
    ]);
    this.status.set('title', exp.title);

    // viewport
    this.sm = new SceneManager(R.viewport);
    this.view = exp.createView(this.sm, this);
    this.banner = el('div', { class: 'banner', role: 'status' });
    R.viewport.append(
      el('div', { class: 'legend' }, (exp.legend || []).map(([color, text]) =>
        el('span', { class: 'legend-item' }, el('i', { style: `background:${color}` }), text))),
      el('div', { class: 'hint' }, 'drag to rotate · scroll to zoom · right-drag to pan'),
      this.banner);

    // plots
    this.plots = exp.plots.map((spec) => {
      const div = el('div', { class: 'plot' });
      R.plots.append(div);
      return { spec, plot: new Plot(div, baseLayout({ title: spec.title, x: spec.x, y: spec.y, extra: spec.layout })) };
    });

    // inspector
    this.pCtl = {}; this.icCtl = {};
    const paramRows = exp.params.map((s) => {
      const c = numberControl(s, (v) => { this.p[s.key] = v; this.onParamChange(); });
      this.pCtl[s.key] = c;
      return c.root;
    });
    const icSliders = [], icCompact = [];
    for (const s of exp.ic) {
      const c = numberControl(s, (v) => { this.ic[s.key] = v; this.reset(); });
      this.icCtl[s.key] = c;
      (s.slider === false ? icCompact : icSliders).push(c.root);
    }
    this.dtCtl = numberControl({
      key: 'dt', label: 'd<i>t</i>', value: this.dt, min: exp.numerics.dtMin, max: exp.numerics.dtMax, log: true,
      aria: 'time step',
    }, (v) => { this.dt = v; this.reinit(); });
    this.methodCtl = selectControl('method', exp.numerics.methods.map((m) => [m, METHOD_NAMES[m]]), this.method,
      (m) => { this.method = m; this.reinit(); });
    this.notes = el('ul', { class: 'notes' });
    this.readouts = readouts();

    const panels = [];
    if (exp.presets?.length) {
      const sel = selectControl('preset', exp.presets.map((ps, i) => [String(i), ps.name]), '0',
        (i) => this.applyPreset(exp.presets[Number(i)]));
      panels.push(section('Presets', sel.root));
    }
    panels.push(
      section('Parameters', ...paramRows),
      section('Initial conditions', ...icSliders,
        icCompact.length ? el('div', { class: 'ctl-grid' }, ...icCompact) : null,
        exp.icNote ? el('p', { class: 'note' }, exp.icNote) : null),
      section('Numerics', this.methodCtl.root, this.dtCtl.root, this.notes),
      section('Readouts', this.readouts.root),
      section('Equations', el('div', { class: 'equations', html: exp.tex.map((t) => tex(t, true)).join('') }),
        exp.blurb ? el('p', { class: 'note' }, exp.blurb) : null),
    );
    R.inspector.append(...panels);

    // transport
    this.playBtn = button('Pause', 'Pause or resume (Space)', () => this.toggle(), 'primary');
    this.speedLabel = el('span', { class: 'speed-value' }, '1×');
    const speed = el('input', { type: 'range', min: -3, max: 5, step: 0.25, value: 0, class: 'speed', 'aria-label': 'simulation speed' });
    speed.addEventListener('input', () => {
      this.speed = 2 ** parseFloat(speed.value);
      this.speedLabel.textContent = `${fmt(this.speed, 3)}×`;
    });
    R.transport.append(
      button('Reset', 'Restart from the initial conditions (R)', () => this.reset()),
      button('Step', 'Advance one integrator step (.)', () => this.singleStep()),
      this.playBtn,
      el('label', { class: 'speed-wrap' }, 'speed', speed, this.speedLabel),
      el('span', { class: 'transport-note' },
        `1× = ${exp.numerics.rate} ${exp.numerics.timeUnit || 'time units'} per second`));

    this.onKey = (e) => {
      if (e.target.matches('input, select, textarea')) return;
      if (e.code === 'Space') { e.preventDefault(); this.toggle(); }
      else if (e.key === 'r' || e.key === 'R') this.reset();
      else if (e.key === '.') this.singleStep();
    };
    window.addEventListener('keydown', this.onKey);
  }

  applyPreset(ps) {
    for (const [k, v] of Object.entries(ps.p || {})) { this.p[k] = v; this.pCtl[k].set(v); }
    for (const [k, v] of Object.entries(ps.ic || {})) { this.ic[k] = v; this.icCtl[k].set(v); }
    if (ps.dt) { this.dt = ps.dt; this.dtCtl.set(ps.dt); }
    if (ps.method) { this.method = ps.method; this.methodCtl.set(ps.method); }
    this.reset();
  }

  /** Update control widgets after the experiment changed values itself. */
  syncControls() {
    for (const [k, c] of Object.entries(this.pCtl)) c.set(this.p[k]);
    for (const [k, c] of Object.entries(this.icCtl)) c.set(this.ic[k]);
  }

  // ---------------------------------------------------------------- engine

  initMessage(runs, t0) {
    return {
      type: 'init', systemId: this.exp.system, runs, method: this.method, dt: this.dt, t0,
      strobe: this.exp.strobe ? this.exp.strobe(this.p) : null,
    };
  }

  /** Restart every run from the initial conditions. */
  async reset() {
    const gen = ++this.gen;
    this.ready = false;
    let runs;
    try {
      const params = this.exp.coreParams(this.p);
      runs = this.exp.initialStates(this.p, this.ic).map((y0) => ({ params, y0: Array.from(y0) }));
    } catch (e) { return this.fail(e); }
    const res = await this.client.call(this.initMessage(runs, 0));
    if (gen !== this.gen) return;
    if (res.error) return this.fail(res.error);
    this.error = null;
    this.n = res.n; this.nRuns = runs.length; this.w = this.n * this.nRuns;
    this.t = 0; this.steps = 0; this.acc = 0;
    this.states = res.states; this.reports = res.reports;
    this.hist = new History(this.w);
    this.hist.push(0, res.states, 0);
    this.section = [];
    this.view.reset?.(res.states, this);
    this.exp.onReset?.(this);
    this.view.append?.(1, res.states, this);
    this.view.state?.(res.states, this);
    this.ready = true; this.dirty = true;
    this.afterChange();
  }

  /** Apply new parameters, step size or method; the trajectory continues from its current state. */
  async reinit() {
    if (this.error || !this.ready) return this.reset();
    const gen = ++this.gen;
    this.ready = false;
    let runs;
    try {
      const params = this.exp.coreParams(this.p);
      runs = Array.from({ length: this.nRuns }, (_, r) => ({
        params, y0: Array.from(this.states.subarray(r * this.n, (r + 1) * this.n)),
      }));
    } catch (e) { return this.fail(e); }
    const res = await this.client.call(this.initMessage(runs, this.t));
    if (gen !== this.gen) return;
    if (res.error) return this.fail(res.error);
    this.reports = res.reports;
    this.view.params?.(this.p, this);
    this.exp.onParams?.(this);
    this.ready = true; this.dirty = true;
    this.afterChange();
  }

  onParamChange() {
    if (this.exp.paramsRestart) this.reset(); else this.reinit();
  }

  fail(e) {
    this.error = `${e.name || 'Error'}: ${e.message}`;
    this.ready = false;
    this.dirty = true;
    this.afterChange();
  }

  afterChange() {
    this.playBtn.textContent = this.running ? 'Pause' : 'Resume';
    this.updateStatus();
  }

  toggle() { this.running = !this.running; this.afterChange(); }

  singleStep() {
    if (this.error || !this.ready || this.pending) return;
    this.running = false;
    this.afterChange();
    this.request(1);
  }

  request(steps) {
    this.pending = true;
    const gen = this.gen, sent = performance.now();
    this.client.call({
      type: 'advance', steps, sampleEvery: Math.ceil(steps / SAMPLES_PER_REQUEST), budgetMs: 10,
    }).then((res) => {
      this.pending = false;
      if (gen !== this.gen) return;
      this.onChunk(res, steps, sent);
    });
  }

  onChunk(res, requested) {
    if (res.error && res.rows === undefined) return this.fail(res.error);
    if (res.done < requested) this.acc = 0; // compute budget hit: drop the backlog instead of spiralling
    const dtSim = res.t - this.t;
    this.rateT += dtSim;
    for (let r = 0; r < res.rows; r++) this.hist.push(res.times[r], res.data, r * this.w);
    this.t = res.t;
    this.steps += res.done;
    this.states = res.states; this.reports = res.reports;
    if (res.section.length) {
      for (const pt of res.section) this.section.push(pt);
      if (this.section.length > 6000) this.section.splice(0, this.section.length - 6000);
      this.view.section?.(res.section, this);
    }
    this.exp.onSamples?.(this, res);
    this.view.append?.(res.rows, res.data, this);
    this.view.state?.(res.states, this);
    this.dirty = true;
    if (res.error) this.fail(res.error);
  }

  // ---------------------------------------------------------------- frame loop

  frame(now) {
    this.raf = requestAnimationFrame(this.frame);
    const wall = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    if (this.running && this.ready && !this.error) {
      this.acc += wall * this.speed * this.exp.numerics.rate;
      this.rateWall += wall;
      if (!this.pending) {
        let steps = Math.floor(this.acc / this.dt);
        if (steps > 0) {
          if (steps > MAX_STEPS_PER_REQUEST) { steps = MAX_STEPS_PER_REQUEST; this.acc = 0; } else this.acc -= steps * this.dt;
          this.request(steps);
        }
      }
      if (this.rateWall >= 0.5) {
        this.achieved = this.rateT / this.rateWall / this.exp.numerics.rate;
        this.rateT = 0; this.rateWall = 0;
      }
    }
    this.sm.render();
    if (this.dirty && now - this.lastPlot > 120) {
      this.lastPlot = now; this.dirty = false;
      this.updatePlots();
      this.updateStatus();
    }
  }

  updatePlots() {
    if (this.hist) {
      for (const { spec, plot } of this.plots) plot.update(spec.traces(this));
      this.readouts.update(this.exp.readouts(this));
    }
    const warnings = new Set();
    for (const r of this.reports) for (const w of r.warnings) warnings.add(w);
    const items = [...warnings];
    if (this.notesKey !== items.join('|')) {
      this.notesKey = items.join('|');
      this.notes.replaceChildren(...items.map((w) => el('li', {}, w)));
    }
  }

  updateStatus() {
    const s = this.status, rep = this.reports[0];
    if (this.error) s.set('state', 'ERROR', 'bad');
    else s.set('state', this.running ? 'RUNNING' : 'PAUSED', this.running ? 'good' : 'idle');
    s.set('t', fmt(this.t, 6));
    s.set('steps', String(this.steps));
    s.set('dt', fmt(this.dt, 4));
    s.set('method', `${METHOD_NAMES[this.method]} · order ${integrators[this.method].order}`);
    const lagging = this.running && this.achieved > 0 && this.achieved < 0.9 * this.speed;
    s.set('speed', this.running && !this.error ? `${fmt(this.achieved, 3)}× of ${fmt(this.speed, 3)}×` : `${fmt(this.speed, 3)}×`,
      lagging ? 'warn' : '');
    if (rep) {
      const z = rep.stability.maxDtRate / rep.stability.limit;
      s.set('stab', fmt(z, 3), z > 1 ? 'bad' : z > 0.5 ? 'warn' : 'good');
    }
    s.set('engine', this.client.mode);
    this.banner.textContent = this.error
      ? `${this.error} — change the step size, method or parameters, or press Reset.` : '';
    this.banner.className = `banner${this.error ? ' show' : ''}`;
  }

  /** State of run r as a subarray of the latest states. */
  state(r = 0) { return this.states.subarray(r * this.n, (r + 1) * this.n); }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.gen++;
    window.removeEventListener('keydown', this.onKey);
    for (const { plot } of this.plots) plot.dispose();
    this.view.dispose?.();
    this.sm.dispose();
  }
}

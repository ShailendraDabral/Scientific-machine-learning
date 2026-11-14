import { systems, createStepper } from '../core/index.js';

/**
 * Simulation engine: owns the steppers and advances them. It contains no
 * rendering code and runs unchanged inside a Web Worker or on the main thread.
 *
 * Messages
 *   { type: 'init', systemId, runs: [{ params, y0 }], method, dt, t0, strobe }
 *   { type: 'advance', steps, sampleEvery, budgetMs }
 * Every reply is a plain object; failures come back as { error: { name, message } }.
 */
export class Engine {
  constructor() { this.steppers = null; }

  handle(msg) {
    try {
      if (msg.type === 'init') return this.init(msg);
      if (msg.type === 'advance') return this.advance(msg);
      return { error: { name: 'Error', message: `unknown message ${msg.type}` } };
    } catch (e) {
      if (msg.type === 'init') this.steppers = null;
      return { error: { name: e.name, message: e.message } };
    }
  }

  init({ systemId, runs, method, dt, t0 = 0, strobe = null }) {
    const system = systems[systemId];
    this.steppers = runs.map((r) => createStepper(system, { params: r.params, y0: r.y0, method, dt, t0 }));
    this.n = this.steppers[0].y.length;
    this.strobe = strobe > 0 && Number.isFinite(strobe) ? strobe : null;
    this.prev = new Float64Array(this.n);
    return { ok: true, n: this.n, t: t0, ...this.snapshot() };
  }

  snapshot() {
    const R = this.steppers.length, n = this.n;
    const states = new Float64Array(R * n);
    this.steppers.forEach((s, i) => states.set(s.y, i * n));
    return { states, reports: this.steppers.map((s) => s.report()) };
  }

  /**
   * Advance every run by up to `steps` steps, stopping early when the time
   * budget is spent. Returns sampled rows [run0 state, run1 state, ...] every
   * `sampleEvery` steps, plus stroboscopic section points when enabled.
   */
  advance({ steps, sampleEvery = 1, budgetMs = Infinity }) {
    if (!this.steppers) return { error: { name: 'Error', message: 'engine not initialised' } };
    const S = this.steppers, R = S.length, n = this.n, lead = S[0];
    const maxRows = Math.floor(steps / sampleEvery) + 1;
    const times = new Float64Array(maxRows), data = new Float64Array(maxRows * R * n);
    const section = [];
    const started = performance.now();
    let rows = 0, done = 0, error = null;
    try {
      while (done < steps) {
        // Largest block that ends on a sample boundary; single steps when strobing.
        const toSample = sampleEvery - (lead.steps % sampleEvery);
        const k = this.strobe ? 1 : Math.min(steps - done, toSample);
        if (this.strobe) this.prev.set(lead.y);
        const tPrev = lead.t;
        for (let r = 0; r < R; r++) S[r].advance(k);
        done += k;
        if (this.strobe) {
          const m = Math.floor(lead.t / this.strobe);
          if (m > Math.floor(tPrev / this.strobe)) {
            // Linear interpolation to the exact strobe time m*T inside this step.
            const f = (m * this.strobe - tPrev) / (lead.t - tPrev), y = lead.y, pt = [];
            for (let i = 0; i < n; i++) pt.push(this.prev[i] + f * (y[i] - this.prev[i]));
            section.push(pt);
          }
        }
        if (lead.steps % sampleEvery === 0) {
          times[rows] = lead.t;
          for (let r = 0; r < R; r++) data.set(S[r].y, (rows * R + r) * n);
          rows++;
        }
        if ((done & 63) === 0 && performance.now() - started > budgetMs) break;
      }
    } catch (e) {
      error = { name: e.name, message: e.message };
    }
    return { rows, times, data, done, t: lead.t, stepCount: lead.steps, section, error, ...this.snapshot() };
  }
}

import { Engine } from './engine.js';

/**
 * Main-thread proxy for the simulation engine. Uses a Web Worker; if the
 * worker cannot be started, it falls back to running the same Engine on the
 * main thread, replaying any calls that were waiting.
 */
export class EngineClient {
  constructor() {
    this.pending = new Map();
    this.nextId = 1;
    this.local = null;
    this.mode = 'worker';
    try {
      this.worker = new Worker(new URL('./sim.worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => {
        const p = this.pending.get(e.data.id);
        if (p) { this.pending.delete(e.data.id); p.resolve(e.data.res); }
      };
      this.worker.onerror = () => this.fallback();
    } catch {
      this.fallback();
    }
  }

  fallback() {
    if (this.local) return;
    this.worker?.terminate();
    this.worker = null;
    this.local = new Engine();
    this.mode = 'main thread';
    const waiting = [...this.pending.values()];
    this.pending.clear();
    for (const p of waiting) p.resolve(this.local.handle(p.msg));
  }

  call(msg) {
    if (this.local) return Promise.resolve(this.local.handle(msg));
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, { msg, resolve });
      this.worker.postMessage({ id, msg });
    });
  }
}

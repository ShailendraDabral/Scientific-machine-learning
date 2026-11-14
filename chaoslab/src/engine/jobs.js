import { runAnalysis } from '../core/analysis/index.js';

/**
 * Runs analysis jobs in a dedicated worker so the interface stays responsive.
 * Starting a new job abandons the one in flight (its worker is terminated).
 * Falls back to the main thread if a worker cannot be started.
 */
export class JobClient {
  constructor() { this.worker = null; this.local = false; this.id = 0; this.mode = 'worker'; }

  spawn() {
    try {
      return new Worker(new URL('./analysis.worker.js', import.meta.url), { type: 'module' });
    } catch {
      this.local = true; this.mode = 'main thread';
      return null;
    }
  }

  run(job) {
    const id = ++this.id;
    if (this.busy && this.worker) { this.worker.terminate(); this.worker = null; }
    if (!this.local && !this.worker) this.worker = this.spawn();
    if (this.local) {
      return new Promise((resolve) => setTimeout(() => resolve(id === this.id ? runAnalysis(job) : { stale: true }), 20));
    }
    this.busy = true;
    const worker = this.worker;
    return new Promise((resolve) => {
      worker.onmessage = (e) => { this.busy = false; resolve(e.data.id === this.id ? e.data.res : { stale: true }); };
      worker.onerror = () => {
        // worker failed to load: compute here instead
        this.local = true; this.mode = 'main thread'; this.worker = null; this.busy = false;
        resolve(runAnalysis(job));
      };
      worker.postMessage({ id, job });
    });
  }

  dispose() { this.worker?.terminate(); this.worker = null; this.id++; }
}

import { runAnalysis } from '../core/analysis/index.js';

self.onmessage = (e) => self.postMessage({ id: e.data.id, res: runAnalysis(e.data.job) });

import { systems } from '../systems/index.js';
import { largestLyapunov, divergence } from './lyapunov.js';
import { sampleObservable, autocorrelation, chooseDelay, delayEmbed } from './embedding.js';
import { poincareSection, sections } from './poincare.js';
import { compareMethods, dtStudy } from './compare.js';

export { largestLyapunov, divergence, sampleObservable, autocorrelation, chooseDelay, delayEmbed,
  poincareSection, sections, compareMethods, dtStudy };

/**
 * Job dispatcher used by the analysis worker. A job is plain data:
 * { kind, systemId, params, y0, dt, method, methods, duration, ... }.
 * Returns plain data, or { error } when the core rejects the request.
 */
export function runAnalysis(job) {
  const started = performance.now();
  try {
    const system = systems[job.systemId];
    let result;
    if (job.kind === 'lyapunov') {
      result = {
        lyapunov: largestLyapunov(system, job),
        divergence: divergence(system, { ...job, duration: job.divergenceDuration }),
      };
    } else if (job.kind === 'embedding') {
      const s = sampleObservable(system, job);
      const acf = autocorrelation(s.x, Math.min(Math.floor(s.x.length / 4), 600));
      const auto = chooseDelay(acf);
      const lag = Number.isInteger(job.lag) && job.lag > 0 ? job.lag : auto.lag;
      const stride = Math.max(1, Math.ceil(s.x.length / 6000));
      const thin = (arr) => arr.filter((_, i) => i % stride === 0);
      const cols = delayEmbed(s.x, lag, 3).map(thin);
      const other = [];
      for (let i = 0; i < s.x.length; i += stride) other.push(s.full[i * s.n + job.index2]);
      result = { cols, acf: Array.from(acf), lag, autoLag: auto.lag, rule: job.lag ? 'set by the user' : auto.rule,
        sampleDt: s.sampleDt, samples: s.x.length, x: thin(Array.from(s.x)), other };
    } else if (job.kind === 'poincare') result = poincareSection(system, job);
    else if (job.kind === 'compare') result = compareMethods(system, job);
    else if (job.kind === 'dtStudy') result = dtStudy(system, job);
    else throw new Error(`unknown analysis "${job.kind}"`);
    return { result, ms: performance.now() - started };
  } catch (e) {
    return { error: { name: e.name, message: e.message }, ms: performance.now() - started };
  }
}

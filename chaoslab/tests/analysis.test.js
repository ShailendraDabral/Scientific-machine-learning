import test from 'node:test';
import assert from 'node:assert/strict';
import {
  lorenz, rossler, duffing, doublePendulum, threeBody, threeBodyPresets,
  largestLyapunov, divergence, autocorrelation, chooseDelay, delayEmbed,
  poincareSection, compareMethods, dtStudy, runAnalysis,
} from '../src/core/index.js';
import { oscillator } from './helpers.js';

test('Lyapunov: Lorenz largest exponent agrees with the published 0.906', () => {
  const r = largestLyapunov(lorenz, { y0: [5, 5, 5], dt: 0.005, transient: 20, duration: 1000, renormTime: 0.1 });
  assert.ok(Math.abs(r.lambda - 0.906) < 0.03, `lambda = ${r.lambda} +- ${r.stderr}`);
  assert.ok(r.stderr > 0 && r.stderr < 0.05);
});

test('Lyapunov: Rossler largest exponent agrees with the published 0.071', () => {
  const r = largestLyapunov(rossler, { dt: 0.01, transient: 200, duration: 6000, renormTime: 0.5 });
  assert.ok(Math.abs(r.lambda - 0.0714) < 0.01, `lambda = ${r.lambda} +- ${r.stderr}`);
});

test('Lyapunov: a harmonic oscillator has exponent zero; a stable Lorenz equilibrium is negative', () => {
  const h = largestLyapunov(oscillator, { dt: 0.01, duration: 500, renormTime: 0.5 });
  assert.ok(Math.abs(h.lambda) < 5e-3, `oscillator ${h.lambda}`);
  const s = largestLyapunov(lorenz, { params: { rho: 14 }, y0: [5, 5, 5], dt: 0.005, transient: 50, duration: 200, renormTime: 0.1 });
  assert.ok(s.lambda < -0.1, `rho = 14: ${s.lambda}`);
});

test('Sensitivity: Lorenz separation grows by orders of magnitude, oscillator separation does not', () => {
  const d = divergence(lorenz, { y0: [5, 5, 5], dt: 0.005, transient: 20, duration: 30, deltas: [1e-9] });
  const s = d.series[0].log10d;
  assert.ok(s.at(-1) - s[0] > 8, `grew by 10^${s.at(-1) - s[0]}`);
  const o = divergence(oscillator, { dt: 0.01, duration: 30, deltas: [1e-9] }).series[0].log10d;
  assert.ok(Math.max(...o) - o[0] < 0.5);
});

test('Embedding: autocorrelation and delay of a sine wave are the analytic ones', () => {
  const N = 4000, x = Float64Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * i) / 100));
  const acf = autocorrelation(x, 200);
  assert.ok(Math.abs(acf[0] - 1) < 1e-12 && Math.abs(acf[50] + 1) < 0.05 && Math.abs(acf[25]) < 0.02);
  // cos(2 pi k / 100) < 1/e first at k = 20 (cos(2 pi 19/100) = 0.3681 > 1/e)
  assert.equal(chooseDelay(acf).lag, 20);
  const cols = delayEmbed(x, 25, 3);
  assert.equal(cols.length, 3); assert.equal(cols[0].length, N - 50);
  assert.equal(cols[1][0], x[25]);
  // sin and its quarter-period shift trace a circle
  assert.ok(Math.abs(cols[0][10] ** 2 + cols[1][10] ** 2 - 1) < 1e-9);
});

test('Poincare: Lorenz points lie on the plane; Duffing strobe gives one point per period', () => {
  const l = poincareSection(lorenz, { y0: [5, 5, 5], dt: 0.005, transient: 20, duration: 100 });
  assert.ok(l.applicable && l.points > 50 && l.residual < 1e-9, `${l.points} points, residual ${l.residual}`);
  const T = (2 * Math.PI) / 1.2;
  const d = poincareSection(duffing, { dt: T / 500, transient: 0, duration: 100 * T - T / 1000 });
  assert.equal(d.points, 99);
  assert.equal(poincareSection(threeBody, { dt: 0.001, duration: 1 }).applicable, false);
});

test('Poincare: crossing of an oscillator is located to interpolation accuracy', () => {
  // x = cos t crosses... use the pendulum section on a linear problem instead: Duffing linear, period-1 response
  const p = { alpha: 1, beta: 0, delta: 0.3, gamma: 0.5, omega: 1.2 };
  const T = (2 * Math.PI) / 1.2;
  const coarse = poincareSection(duffing, { params: p, y0: [0, 0], dt: T / 37.3, transient: 40 * T, duration: 20 * T });
  // steady state is periodic with the drive: every strobe point is the same
  const spread = Math.max(...coarse.a) - Math.min(...coarse.a);
  assert.ok(spread < 1e-4, `spread ${spread}`);
});

test('Comparison: errors rank by order, symplectic methods bound energy, inapplicable methods are refused', () => {
  const f = threeBodyPresets.figureEight();
  const c = compareMethods(threeBody, { ...f, dt: 0.002, duration: 6, methods: ['euler', 'rk4', 'symplecticEuler', 'verlet'] });
  const by = Object.fromEntries(c.results.map((r) => [r.method, r]));
  assert.ok(by.rk4.maxErr < by.verlet.maxErr && by.verlet.maxErr < by.euler.maxErr);
  assert.ok(by.verlet.maxEnergyErr < 1e-4 && by.euler.maxEnergyErr > 10 * by.symplecticEuler.maxEnergyErr);
  assert.ok(c.reference.selfDifference < by.rk4.maxErr);
  assert.ok(c.results.every((r) => r.nsPerStep > 0 && r.status === 'stable'));
  const dp = compareMethods(doublePendulum, { dt: 0.002, duration: 2, methods: ['rk4', 'verlet'] });
  assert.equal(dp.results[1].applicable, false);
  const bad = compareMethods(lorenz, { y0: [5, 5, 5], dt: 0.05, duration: 10, methods: ['euler', 'rk4'] });
  assert.equal(bad.results[0].status, 'diverged');
  assert.equal(bad.results[0].nsPerStep, null);
});

test('Time-step study: observed orders match the nominal orders', () => {
  const s = dtStudy(lorenz, { y0: [5, 5, 5], dt: 0.01, duration: 1, methods: ['euler', 'midpoint', 'rk4'] });
  const o = Object.fromEntries(s.results.map((r) => [r.method, r.order]));
  assert.ok(Math.abs(o.euler - 1) < 0.25, `euler ${o.euler}`);
  assert.ok(Math.abs(o.midpoint - 2) < 0.2, `midpoint ${o.midpoint}`);
  assert.ok(Math.abs(o.rk4 - 4) < 0.3, `rk4 ${o.rk4}`);
  const f = threeBodyPresets.figureEight();
  const h = dtStudy(threeBody, { ...f, dt: 0.02, duration: 2, methods: ['symplecticEuler', 'verlet'] });
  assert.ok(Math.abs(h.results[0].order - 1) < 0.2 && Math.abs(h.results[1].order - 2) < 0.2,
    `${h.results[0].order} ${h.results[1].order}`);
});

test('Job dispatcher returns plain data and reports core errors', () => {
  const ok = runAnalysis({ kind: 'embedding', systemId: 'lorenz', y0: [5, 5, 5], dt: 0.005, transient: 20, duration: 40, sampleDt: 0.02, index: 0, index2: 2 });
  assert.ok(ok.result.cols[0].length > 100 && ok.result.lag >= 1);
  assert.doesNotThrow(() => structuredClone(ok));
  const bad = runAnalysis({ kind: 'compare', systemId: 'lorenz', params: { sigma: -1 }, dt: 0.01, duration: 1, methods: ['rk4'] });
  assert.equal(bad.error.name, 'InvalidParameterError');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { integrate, integrators, InvalidParameterError, NumericalInstabilityError, lorenz, doublePendulum, duffing } from '../src/core/index.js';
import { oscillator } from './helpers.js';

/** Global error at t = 1 for x'' = -x, x(0) = 1, v(0) = 0. */
function errorAt(method, steps) {
  const r = integrate(oscillator, { method, dt: 1 / steps, steps });
  return Math.hypot(r.final[0] - Math.cos(1), r.final[1] + Math.sin(1));
}

for (const [id, m] of Object.entries(integrators)) {
  test(`${m.name}: observed convergence order equals ${m.order}`, () => {
    const e1 = errorAt(id, 200), e2 = errorAt(id, 400);
    const order = Math.log2(e1 / e2);
    assert.ok(Math.abs(order - m.order) < 0.1, `observed order ${order.toFixed(3)}`);
  });
}

test('symplectic methods keep energy bounded; Euler does not', () => {
  const run = (method) => integrate(oscillator, { method, dt: 0.05, steps: 20000 }).invariants.energy.maxRelDrift;
  assert.ok(run('verlet') < 1e-3);
  assert.ok(run('symplecticEuler') < 0.06);
  assert.ok(run('euler') > 10); // energy grows by (1 + dt^2) every step
  assert.ok(run('rk4') < 1e-5);
});

test('Stormer-Verlet is time-reversible', () => {
  const fwd = integrate(oscillator, { method: 'verlet', dt: 0.01, steps: 1000 });
  const back = integrate(oscillator, { method: 'verlet', dt: 0.01, steps: 1000, y0: [fwd.final[0], -fwd.final[1]] });
  assert.ok(Math.abs(back.final[0] - 1) < 1e-12 && Math.abs(back.final[1]) < 1e-12);
});

test('symplectic methods are refused for non-separable systems', () => {
  assert.throws(() => integrate(doublePendulum, { method: 'verlet', dt: 0.001, steps: 1 }), InvalidParameterError);
  assert.throws(() => integrate(lorenz, { method: 'symplecticEuler', dt: 0.001, steps: 1 }), InvalidParameterError);
  assert.throws(() => integrate(duffing, { method: 'verlet', dt: 0.001, steps: 1 }), /not Hamiltonian/);
});

test('solver rejects invalid options', () => {
  for (const dt of [0, -0.1, NaN, Infinity, '0.1', undefined]) {
    assert.throws(() => integrate(lorenz, { dt, steps: 1 }), InvalidParameterError);
  }
  assert.throws(() => integrate(lorenz, { dt: 0.01, steps: -1 }), InvalidParameterError);
  assert.throws(() => integrate(lorenz, { dt: 0.01, steps: 1.5 }), InvalidParameterError);
  assert.throws(() => integrate(lorenz, { dt: 0.01, steps: 1, method: 'rk5' }), InvalidParameterError);
  assert.throws(() => integrate(lorenz, { dt: 0.01, steps: 1, y0: [1, 2] }), InvalidParameterError);
  assert.throws(() => integrate(lorenz, { dt: 0.01, steps: 1, y0: [1, NaN, 0] }), InvalidParameterError);
});

test('unstable step sizes are detected, not returned', () => {
  // Forward Euler on Lorenz with dt = 0.1 diverges.
  assert.throws(() => integrate(lorenz, { method: 'euler', dt: 0.1, steps: 5000 }), NumericalInstabilityError);
  // strict mode refuses before any damage is done
  assert.throws(() => integrate(lorenz, { method: 'euler', dt: 0.1, steps: 10, strict: true }), NumericalInstabilityError);
  // non-strict run at a marginal step reports a warning
  const r = integrate(lorenz, { method: 'rk4', dt: 0.1, steps: 200 });
  assert.equal(r.stability.ok, false);
  assert.ok(r.warnings.some((w) => w.includes('stability limit')));
  // a safe step is reported clean
  const ok = integrate(lorenz, { method: 'rk4', dt: 0.005, steps: 2000 });
  assert.equal(ok.stability.ok, true);
  assert.equal(ok.warnings.length, 0);
});

test('time is n*dt exactly and sampling shape is correct', () => {
  const r = integrate(lorenz, { dt: 0.1 / 3, steps: 10, sampleEvery: 3 });
  assert.equal(r.rows, 4);
  assert.equal(r.y.length, 12);
  assert.equal(r.t[3], 9 * (0.1 / 3));
  assert.equal(r.tFinal, 10 * (0.1 / 3));
  assert.equal(r.steps, 10);
});

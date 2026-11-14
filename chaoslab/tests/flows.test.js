import test from 'node:test';
import assert from 'node:assert/strict';
import { integrate, lorenz, rossler, duffing, InvalidParameterError, NumericalInstabilityError } from '../src/core/index.js';
import { fdJacobian, maxAbsDiff } from './helpers.js';

test('Lorenz: right-hand side matches the equations', () => {
  const { params: p } = lorenz.resolveParams({ sigma: 10, rho: 28, beta: 8 / 3 });
  const d = new Float64Array(3);
  lorenz.rhs(0, [1, 2, 3], d, p);
  assert.deepEqual(Array.from(d), [10 * (2 - 1), 1 * (28 - 3) - 2, 1 * 2 - (8 / 3) * 3]);
});

test('Lorenz: equilibria are zeros of the vector field', () => {
  const { params: p } = lorenz.resolveParams();
  const d = new Float64Array(3);
  const eq = lorenz.equilibria(p);
  assert.equal(eq.length, 3);
  for (const e of eq) {
    lorenz.rhs(0, e, d, p);
    assert.ok(Math.max(...d.map(Math.abs)) < 1e-12);
  }
});

test('Lorenz: analytic Jacobian agrees with finite differences; trace is -(sigma+1+beta)', () => {
  const { params: p } = lorenz.resolveParams();
  const y = [-3.2, 4.1, 21.7], J = new Float64Array(9);
  lorenz.jacobian(0, y, J, p);
  assert.ok(maxAbsDiff(J, fdJacobian(lorenz, 0, y, p)) < 1e-7);
  assert.ok(Math.abs(J[0] + J[4] + J[8] - lorenz.divergence(p)) < 1e-14);
});

test('Lorenz: RK4 solution converges at fourth order', () => {
  const run = (steps) => integrate(lorenz, { dt: 1 / steps, steps }).final;
  const a = run(2000), b = run(4000), c = run(8000);
  const order = Math.log2(maxAbsDiff(a, b) / maxAbsDiff(b, c));
  assert.ok(Math.abs(order - 4) < 0.2, `observed order ${order}`);
});

test('Lorenz: trajectory stays on the bounded attractor', () => {
  const r = integrate(lorenz, { dt: 0.005, steps: 40000, sampleEvery: 10 });
  let zmin = Infinity, zmax = -Infinity, xmax = 0;
  for (let i = 200; i < r.rows; i++) { // skip transient
    const x = r.y[3 * i], z = r.y[3 * i + 2];
    zmin = Math.min(zmin, z); zmax = Math.max(zmax, z); xmax = Math.max(xmax, Math.abs(x));
  }
  assert.ok(zmin > 0 && zmax < 50 && xmax < 22 && xmax > 15, `${zmin} ${zmax} ${xmax}`);
});

test('Lorenz: invalid parameters are rejected', () => {
  for (const bad of [{ sigma: 0 }, { sigma: -1 }, { beta: 0 }, { rho: -5 }, { rho: NaN }, { sigma: '10' }, { tau: 1 }]) {
    assert.throws(() => lorenz.resolveParams(bad), InvalidParameterError, JSON.stringify(bad));
  }
  assert.throws(() => lorenz.resolveParams(null), InvalidParameterError);
});

test('Rossler: equilibria, Jacobian and boundedness', () => {
  const { params: p } = rossler.resolveParams();
  const d = new Float64Array(3);
  const eq = rossler.equilibria(p);
  assert.equal(eq.length, 2);
  for (const e of eq) {
    rossler.rhs(0, e, d, p);
    assert.ok(Math.max(...d.map(Math.abs)) < 1e-10);
  }
  const y = [2.5, -1.5, 0.7], J = new Float64Array(9);
  rossler.jacobian(0, y, J, p);
  assert.ok(maxAbsDiff(J, fdJacobian(rossler, 0, y, p)) < 1e-7);
  const r = integrate(rossler, { dt: 0.01, steps: 50000 });
  assert.ok(Math.abs(r.final[0]) < 15 && Math.abs(r.final[1]) < 15 && r.final[2] >= 0 && r.final[2] < 30);
  assert.equal(r.warnings.length, 0);
});

test('Rossler: escaping trajectories raise an error; risky parameters warn', () => {
  assert.throws(() => integrate(rossler, { dt: 0.01, steps: 100000, y0: [0, 0, 500] , maxNorm: 1e8}), NumericalInstabilityError);
  assert.ok(rossler.resolveParams({ a: 2, b: 2, c: 1 }).warnings.length > 0);
});

test('Duffing: linear limit reproduces the exact steady-state amplitude', () => {
  const params = { alpha: 1, beta: 0, delta: 0.3, gamma: 0.5, omega: 1.2 };
  const T = (2 * Math.PI) / params.omega, N = 500;
  const r = integrate(duffing, { params, dt: T / N, steps: 60 * N, y0: [0, 0] });
  let amp = 0;
  for (let i = 55 * N; i < r.rows; i++) amp = Math.max(amp, Math.abs(r.y[2 * i]));
  const exact = params.gamma / Math.hypot(params.alpha - params.omega ** 2, params.delta * params.omega);
  assert.ok(Math.abs(amp / exact - 1) < 1e-4, `${amp} vs ${exact}`);
});

test('Duffing: undamped, undriven motion conserves energy (Verlet bounded, RK4 tiny)', () => {
  const params = { alpha: -1, beta: 1, delta: 0, gamma: 0 };
  const v = integrate(duffing, { params, method: 'verlet', dt: 0.01, steps: 100000, y0: [0.5, 0.8] });
  assert.ok(v.invariants.energy.maxRelDrift < 1e-3);
  const r = integrate(duffing, { params, method: 'rk4', dt: 0.01, steps: 100000, y0: [0.5, 0.8] });
  assert.ok(r.invariants.energy.maxRelDrift < 1e-6);
  // with damping, energy is not an invariant and is not reported
  assert.deepEqual(integrate(duffing, { dt: 0.01, steps: 10 }).invariants, {});
});

test('Duffing: chaotic default stays bounded; Jacobian is correct; bad parameters rejected', () => {
  const T = duffing.drivePeriod(duffing.resolveParams().params);
  const r = integrate(duffing, { dt: T / 1000, steps: 200000 });
  assert.ok(Math.abs(r.final[0]) < 2 && Math.abs(r.final[1]) < 2);
  const { params: p } = duffing.resolveParams();
  const J = new Float64Array(4);
  duffing.jacobian(0.3, [0.7, -0.2], J, p);
  assert.ok(maxAbsDiff(J, fdJacobian(duffing, 0.3, [0.7, -0.2], p)) < 1e-7);
  assert.throws(() => duffing.resolveParams({ delta: -0.1 }), InvalidParameterError);
  assert.throws(() => duffing.resolveParams({ omega: -1 }), InvalidParameterError);
  assert.ok(duffing.resolveParams({ beta: -1 }).warnings.length === 1);
});

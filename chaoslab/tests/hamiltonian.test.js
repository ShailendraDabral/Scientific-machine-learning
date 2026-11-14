import test from 'node:test';
import assert from 'node:assert/strict';
import {
  integrate, doublePendulum, threeBody, threeBodyPresets, toCenterOfMassFrame,
  InvalidParameterError, NumericalInstabilityError,
} from '../src/core/index.js';
import { maxAbsDiff } from './helpers.js';

test('Double pendulum: accelerations satisfy both Euler-Lagrange equations', () => {
  const cases = [
    [{ m1: 1, m2: 1, l1: 1, l2: 1, g: 9.81 }, [0.3, -1.1, 2.0, -0.7]],
    [{ m1: 2.5, m2: 0.4, l1: 0.7, l2: 1.9, g: 9.81 }, [2.8, 1.3, -4.2, 5.5]],
    [{ m1: 0.2, m2: 3.0, l1: 1.4, l2: 0.3, g: 3.7 }, [-1.9, 3.05, 0.6, -8.0]],
  ];
  for (const [params, y] of cases) {
    const { params: p } = doublePendulum.resolveParams(params);
    const d = new Float64Array(4);
    doublePendulum.rhs(0, y, d, p);
    const [th1, th2, w1, w2] = y, a1 = d[2], a2 = d[3], D = th1 - th2;
    const { m1, m2, l1, l2, g } = p;
    // d/dt(dL/dw) - dL/dtheta = 0, divided by l1 and by m2*l2 respectively
    const e1 = (m1 + m2) * l1 * a1 + m2 * l2 * a2 * Math.cos(D) + m2 * l2 * w2 * w2 * Math.sin(D) + (m1 + m2) * g * Math.sin(th1);
    const e2 = l2 * a2 + l1 * a1 * Math.cos(D) - l1 * w1 * w1 * Math.sin(D) + g * Math.sin(th2);
    assert.ok(Math.abs(e1) < 1e-10 && Math.abs(e2) < 1e-10, `${e1} ${e2}`);
    assert.equal(d[0], w1); assert.equal(d[1], w2);
  }
});

test('Double pendulum: small oscillations follow the analytic normal modes', () => {
  const g = 9.81;
  for (const [sign, shape] of [[-1, Math.SQRT2], [1, -Math.SQRT2]]) {
    const w = Math.sqrt(g * (2 + sign * Math.SQRT2)), T = (2 * Math.PI) / w, N = 4000;
    const y0 = [1e-3, shape * 1e-3, 0, 0];
    const r = integrate(doublePendulum, { dt: T / N, steps: N, y0 });
    assert.ok(Math.abs(r.final[0] / y0[0] - 1) < 1e-4 && Math.abs(r.final[1] / y0[1] - 1) < 1e-4,
      `mode ${sign}: ${r.final[0]} ${r.final[1]}`);
  }
});

test('Double pendulum: RK4 conserves energy to 1e-6 over 100 s of chaotic motion', () => {
  const r = integrate(doublePendulum, { dt: 0.001, steps: 100000 });
  assert.ok(r.invariants.energy.maxRelDrift === null || r.invariants.energy.maxRelDrift < 1e-6);
  // default release has E = 0 exactly (both rods horizontal), so check absolute drift too
  assert.ok(r.invariants.energy.maxAbsDrift < 1e-5, `${r.invariants.energy.maxAbsDrift}`);
  const lowOrder = integrate(doublePendulum, { method: 'euler', dt: 0.001, steps: 20000 });
  assert.ok(lowOrder.invariants.energy.maxAbsDrift > 1e-2); // Euler visibly gains energy
});

test('Double pendulum: invalid parameters are rejected', () => {
  for (const bad of [{ m1: 0 }, { m2: -1 }, { l1: 0 }, { l2: -2 }, { g: -9.81 }, { m1: Infinity }]) {
    assert.throws(() => doublePendulum.resolveParams(bad), InvalidParameterError, JSON.stringify(bad));
  }
});

test('Three-body: forces obey Newton III and the two-body circular orbit closes', () => {
  const { params: p } = threeBody.resolveParams({ masses: [3, 4, 5] });
  const y = threeBodyPresets.pythagorean().y0, a = new Float64Array(6);
  threeBody.separable.accel(0, y, a, p);
  // sum of m_i a_i vanishes
  assert.ok(Math.abs(3 * a[0] + 4 * a[2] + 5 * a[4]) < 1e-14 && Math.abs(3 * a[1] + 4 * a[3] + 5 * a[5]) < 1e-14);
  // body 0 at (1,3): pulled by m=4 at distance 5 and m=5 at distance 4
  assert.ok(Math.abs(a[0] - 4 * (-3) / 125) < 1e-15 && Math.abs(a[1] - (4 * (-4) / 125 + 5 * (-4) / 64)) < 1e-15);

  // Two unit masses, separation 1: Omega^2 = G (m1 + m2) / d^3 = 2
  const Om = Math.SQRT2, T = (2 * Math.PI) / Om, N = 4000;
  const y0 = [0.5, 0, -0.5, 0, 0, 0.5 * Om, 0, -0.5 * Om];
  const r = integrate(threeBody, { params: { masses: [1, 1] }, dt: T / N, steps: N, y0 });
  assert.ok(maxAbsDiff(r.final, y0) < 1e-9);
});

test('Three-body: figure-eight orbit returns to its initial state after one period', () => {
  const { params, y0, period } = threeBodyPresets.figureEight();
  const N = 20000;
  const r = integrate(threeBody, { params, y0, dt: period / N, steps: N });
  assert.ok(maxAbsDiff(r.final, y0) < 1e-5, `closure error ${maxAbsDiff(r.final, y0)}`);
  assert.ok(r.invariants.energy.maxRelDrift < 1e-10);
});

test('Three-body: Lagrange equilateral solution rotates rigidly', () => {
  const { params, y0, period } = threeBodyPresets.lagrange();
  const N = 10000;
  const r = integrate(threeBody, { params, y0, dt: period / N, steps: N });
  assert.ok(maxAbsDiff(r.final, y0) < 1e-8);
});

test('Three-body: Verlet bounds energy error and conserves momenta to round-off', () => {
  const { params, y0, period } = threeBodyPresets.figureEight();
  const r = integrate(threeBody, { params, y0, method: 'verlet', dt: 0.001, steps: Math.round((10 * period) / 0.001) });
  const inv = r.invariants;
  assert.ok(inv.energy.maxRelDrift < 1e-5, `E drift ${inv.energy.maxRelDrift}`);
  assert.ok(inv.momentumX.maxAbsDrift < 1e-12 && inv.momentumY.maxAbsDrift < 1e-12);
  assert.ok(inv.angularMomentumZ.maxAbsDrift < 1e-12);
  // halving dt reduces the energy error by ~4 (second order)
  const r2 = integrate(threeBody, { params, y0, method: 'verlet', dt: 0.0005, steps: Math.round((10 * period) / 0.0005) });
  const ratio = inv.energy.maxAbsDrift / r2.invariants.energy.maxAbsDrift;
  assert.ok(ratio > 3.5 && ratio < 4.5, `ratio ${ratio}`);
});

test('Three-body: 3D state, centre-of-mass shift, and symplectic Euler', () => {
  const { params, y0 } = threeBodyPresets.figureEight({ dim: 3 });
  assert.equal(y0.length, 18);
  const r = integrate(threeBody, { params, y0, method: 'symplecticEuler', dt: 0.0005, steps: 4000 });
  assert.ok(r.invariants.energy.maxRelDrift < 1e-2);
  assert.equal(r.invariants.momentumZ.maxAbsDrift, 0);
  const { params: p } = threeBody.resolveParams({ masses: [1, 2, 3] });
  const y = Float64Array.from([1, 0, 0, 1, -1, 0, 0.3, 0, 0, 0.2, -0.1, 0.4]);
  toCenterOfMassFrame(y, p);
  const inv = threeBody.invariants(p);
  assert.ok(Math.abs(inv.momentumX(0, y)) < 1e-15 && Math.abs(inv.momentumY(0, y)) < 1e-15);
  assert.ok(Math.abs(1 * y[0] + 2 * y[2] + 3 * y[4]) < 1e-15);
});

test('Three-body: close encounters warn, collisions and bad parameters throw', () => {
  const pyth = threeBodyPresets.pythagorean();
  const r = integrate(threeBody, { ...pyth, method: 'verlet', dt: 0.01, steps: 400, maxNorm: 1e6 });
  assert.equal(typeof r.stability.maxDtRate, 'number');
  assert.throws(() => integrate(threeBody, { params: { masses: [1, 1] }, y0: [0, 0, 0, 0, 0, 0, 0, 0], dt: 0.01, steps: 1 }),
    NumericalInstabilityError);
  for (const bad of [{ masses: [1, 0, 1] }, { masses: [1, -1, 1] }, { masses: [1] }, { G: 0 }, { G: -1 },
    { softening: -0.1 }, { dim: 4 }, { masses: 'abc' }, { mass: 1 }]) {
    assert.throws(() => threeBody.resolveParams(bad), InvalidParameterError, JSON.stringify(bad));
  }
  assert.ok(threeBody.resolveParams({ softening: 0.01 }).warnings.length === 1);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { logistic, InvalidParameterError } from '../src/core/index.js';

test('Logistic: single iterate is exact', () => {
  assert.equal(logistic.map(0.25, { r: 4 }), 0.75);
  assert.equal(logistic.orbit({ params: { r: 2 }, x0: 0.25, n: 3 })[1], 0.375);
});

test('Logistic: converges to the fixed point 1 - 1/r for 1 < r < 3', () => {
  const o = logistic.orbit({ params: { r: 2.5 }, n: 1, discard: 200 });
  assert.ok(Math.abs(o[0] - 0.6) < 1e-14);
});

test('Logistic: period-2 orbit at r = 3.2 matches the closed form', () => {
  const r = 3.2, s = Math.sqrt((r - 3) * (r + 1));
  const exact = [(r + 1 - s) / (2 * r), (r + 1 + s) / (2 * r)];
  const o = Array.from(logistic.orbit({ params: { r }, n: 2, discard: 2000 })).sort();
  assert.ok(Math.abs(o[0] - exact[0]) < 1e-12 && Math.abs(o[1] - exact[1]) < 1e-12);
});

test('Logistic: Lyapunov exponent is ln 2 at r = 4, negative on cycles, 0 at r = 3 onset', () => {
  assert.ok(Math.abs(logistic.lyapunov({ params: { r: 4 }, n: 1e6 }) - Math.LN2) < 1e-3);
  assert.ok(logistic.lyapunov({ params: { r: 3.2 } }) < -0.5);
  // stable fixed point: lambda = ln|2 - r|
  assert.ok(Math.abs(logistic.lyapunov({ params: { r: 2.5 } }) - Math.log(0.5)) < 1e-9);
  // period-3 window is periodic, r = 3.9 is chaotic
  assert.ok(logistic.lyapunov({ params: { r: 3.835 } }) < 0);
  assert.ok(logistic.lyapunov({ params: { r: 3.9 } }) > 0.4);
});

test('Logistic: orbit never leaves [0, 1] for r in [0, 4]', () => {
  for (const r of [0, 0.5, 1, 2.9, 3.57, 3.9, 4]) {
    const o = logistic.orbit({ params: { r }, x0: 0.123, n: 5000 });
    assert.ok(o.every((x) => x >= 0 && x <= 1), `r = ${r}`);
  }
});

test('Logistic: invalid inputs are rejected', () => {
  assert.throws(() => logistic.orbit({ params: { r: 4.1 }, n: 10 }), InvalidParameterError);
  assert.throws(() => logistic.orbit({ params: { r: -0.1 }, n: 10 }), InvalidParameterError);
  assert.throws(() => logistic.orbit({ params: { r: 3 }, x0: 1.2, n: 10 }), InvalidParameterError);
  assert.throws(() => logistic.orbit({ params: { r: 3 }, x0: NaN, n: 10 }), InvalidParameterError);
  assert.throws(() => logistic.orbit({ params: { r: 3 }, n: -1 }), InvalidParameterError);
  assert.throws(() => logistic.orbit({ params: { r: 3 }, n: 2.5 }), InvalidParameterError);
});

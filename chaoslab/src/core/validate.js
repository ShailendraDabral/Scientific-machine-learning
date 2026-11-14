import { InvalidParameterError } from './errors.js';

/**
 * Validate scalar parameters against a schema and fill in defaults.
 *
 * schema: { key: { default, min?, max?, minExclusive?, maxExclusive?, description } }
 * Returns a new plain object holding every schema key.
 */
export function resolveScalarParams(systemId, schema, input = {}) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new InvalidParameterError(`${systemId}: params must be an object`);
  }
  for (const key of Object.keys(input)) {
    if (!(key in schema)) {
      throw new InvalidParameterError(`${systemId}: unknown parameter "${key}"`, { key });
    }
  }
  const out = {};
  for (const [key, spec] of Object.entries(schema)) {
    const v = input[key] === undefined ? spec.default : input[key];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new InvalidParameterError(
        `${systemId}: parameter "${key}" must be a finite number, got ${String(v)}`, { key, value: v });
    }
    if (spec.min !== undefined && (spec.minExclusive ? v <= spec.min : v < spec.min)) {
      throw new InvalidParameterError(
        `${systemId}: parameter "${key}" = ${v} must be ${spec.minExclusive ? '>' : '>='} ${spec.min}`,
        { key, value: v });
    }
    if (spec.max !== undefined && (spec.maxExclusive ? v >= spec.max : v > spec.max)) {
      throw new InvalidParameterError(
        `${systemId}: parameter "${key}" = ${v} must be ${spec.maxExclusive ? '<' : '<='} ${spec.max}`,
        { key, value: v });
    }
    out[key] = v;
  }
  return out;
}

/** Check that y is an array-like of exactly n finite numbers. */
export function checkState(systemId, y, n) {
  if (y == null || typeof y.length !== 'number') {
    throw new InvalidParameterError(`${systemId}: state must be an array of ${n} numbers`);
  }
  if (y.length !== n) {
    throw new InvalidParameterError(
      `${systemId}: state has length ${y.length}, expected ${n}`, { length: y.length, expected: n });
  }
  for (let i = 0; i < n; i++) {
    if (typeof y[i] !== 'number' || !Number.isFinite(y[i])) {
      throw new InvalidParameterError(
        `${systemId}: state component ${i} is not a finite number (${String(y[i])})`, { index: i });
    }
  }
}

/** Require a finite, strictly positive number (used for dt). */
export function checkPositive(name, v) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
    throw new InvalidParameterError(`${name} must be a finite number > 0, got ${String(v)}`);
  }
}

/** Require a non-negative integer (used for step and iteration counts). */
export function checkCount(name, v) {
  if (!Number.isInteger(v) || v < 0) {
    throw new InvalidParameterError(`${name} must be a non-negative integer, got ${String(v)}`);
  }
}

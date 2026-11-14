import { InvalidParameterError } from '../errors.js';
import { euler } from './euler.js';
import { midpoint } from './midpoint.js';
import { rk4 } from './rk4.js';
import { symplecticEuler } from './symplecticEuler.js';
import { verlet } from './verlet.js';

/**
 * Integrator registry. Every method has the same in-place, allocation-free
 * signature:  step(system, params, t, y, h, out, work)
 * It reads y, writes the new state to out (out must not alias y).
 */
export const integrators = { euler, midpoint, rk4, symplecticEuler, verlet };

export function getIntegrator(id) {
  const m = integrators[id];
  if (!m) {
    throw new InvalidParameterError(
      `unknown integrator "${id}"; available: ${Object.keys(integrators).join(', ')}`);
  }
  return m;
}

export { euler, midpoint, rk4, symplecticEuler, verlet };

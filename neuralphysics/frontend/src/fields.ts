// Display helpers: reshape flat float32 fields into 2D views and map values to colours.
// Nothing here changes a value; it only arranges numbers for drawing.
import type { GridInfo } from './api';

export interface View2D { z: Float32Array[]; x: number[]; y: number[]; xLabel: string; yLabel: string }

/** (x, t) fields plot t horizontally and x vertically; (x, y[, t]) fields plot x horizontally and y vertically. */
export function toView(data: Float32Array, shape: number[], grid: GridInfo, tIndex = 0): View2D {
  if (shape.length === 2 && grid.names[1] === 't') {
    const [nx, nt] = shape;
    const z = Array.from({ length: nx }, (_, i) => data.subarray(i * nt, (i + 1) * nt));
    return { z, x: grid.axes[1], y: grid.axes[0], xLabel: 't', yLabel: 'x' };
  }
  const [nx, ny] = shape;
  const nt = shape[2] ?? 1;
  const k = Math.min(tIndex, nt - 1);
  const z = Array.from({ length: ny }, (_, j) => {
    const row = new Float32Array(nx);
    for (let i = 0; i < nx; i++) row[i] = data[(i * ny + j) * nt + k];
    return row;
  });
  return { z, x: grid.axes[0], y: grid.axes[1], xLabel: 'x', yLabel: 'y' };
}

export function range(data: Float32Array | undefined): [number, number] {
  if (!data || data.length === 0) return [0, 1];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137],
  [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];

export function viridis(t: number): [number, number, number] {
  const s = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(s));
  const f = s - i;
  return [0, 1, 2].map((c) => (VIRIDIS[i][c] * (1 - f) + VIRIDIS[i + 1][c] * f) / 255) as [number, number, number];
}

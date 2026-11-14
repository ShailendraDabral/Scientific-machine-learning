/** Recent sampled rows [t, state of every run]; the oldest quarter is dropped when full. */
export class History {
  constructor(width, capacity = 8000) {
    this.w = width; this.capacity = capacity; this.count = 0;
    this.t = new Float64Array(capacity);
    this.data = new Float64Array(capacity * width);
  }
  clear() { this.count = 0; }
  push(t, src, offset) {
    if (this.count === this.capacity) {
      const drop = this.capacity >> 2;
      this.t.copyWithin(0, drop, this.count);
      this.data.copyWithin(0, drop * this.w, this.count * this.w);
      this.count -= drop;
    }
    this.t[this.count] = t;
    for (let i = 0; i < this.w; i++) this.data[this.count * this.w + i] = src[offset + i];
    this.count++;
  }
  /**
   * Build plot arrays from stored rows. fx and fy receive (t, data, offset)
   * where data[offset + k] is component k of the row.
   */
  xy(fx, fy, { window = Infinity, max = 2000 } = {}) {
    const N = this.count, x = [], y = [];
    if (!N) return { x, y };
    const tMin = this.t[N - 1] - window;
    let start = 0;
    if (Number.isFinite(tMin)) { while (start < N - 1 && this.t[start] < tMin) start++; }
    const stride = Math.max(1, Math.ceil((N - start) / max));
    for (let i = start; i < N; i += stride) {
      x.push(fx(this.t[i], this.data, i * this.w));
      y.push(fy(this.t[i], this.data, i * this.w));
    }
    return { x, y };
  }
}

/** Whole-run series with adaptive thinning: never more than `cap` points. */
export class LongSeries {
  constructor(columns, cap = 2400) { this.k = columns; this.cap = cap; this.clear(); }
  clear() { this.cols = Array.from({ length: this.k }, () => []); this.stride = 1; this.i = 0; }
  push(...vals) {
    if (this.i++ % this.stride) return;
    for (let j = 0; j < this.k; j++) this.cols[j].push(vals[j]);
    if (this.cols[0].length >= this.cap) {
      this.cols = this.cols.map((c) => c.filter((_, i) => i % 2 === 0));
      this.stride *= 2;
    }
  }
  col(j) { return this.cols[j]; }
}

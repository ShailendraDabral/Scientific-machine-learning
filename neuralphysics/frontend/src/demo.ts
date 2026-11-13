// Demo mode: the browser replays recordings of real training runs instead of talking to the
// Python back end. A recording is the exact message stream the API sent for that run
// (see scripts/record_demo.py); nothing is simulated here. Arrays are stored as float16.
import type { Frame, RunEvent } from './api';

interface Message { t: number; e?: RunEvent; f?: { name: string; step: number; shape: number[]; b64: string } }
interface Recording { key: string; kind: string; request: unknown; duration: number; recorded: string; messages: Message[]; new_inputs?: { length_scale: number }[] }

const MAX_REPLAY_SECONDS = 30; // long runs are time-compressed to this; shorter ones play in real time
const cache = new Map<string, Promise<unknown>>();

function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    const inline = document.getElementById(`demo-data-${name}`);
    cache.set(name, inline
      ? Promise.resolve(JSON.parse(inline.textContent ?? 'null'))
      : fetch(`${import.meta.env.BASE_URL}demo/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`No demo recording named ${name}`);
        return r.json();
      }));
  }
  return cache.get(name) as Promise<T>;
}

function float16ToFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const out = new Float32Array(bin.length / 2);
  for (let i = 0; i < out.length; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
    out[i] = exp === 0 ? sign * 2 ** -14 * (frac / 1024)
      : exp === 31 ? (frac ? NaN : sign * Infinity)
      : sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

class Replay {
  paused = false;
  private clock = 0;
  private next = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  constructor(rec: Recording, private onEvent: (e: RunEvent) => void, private onFrame: (f: Frame) => void) {
    const scale = Math.min(1, MAX_REPLAY_SECONDS / rec.duration);
    let last = performance.now();
    this.timer = setInterval(() => {
      const now = performance.now();
      if (!this.paused) this.clock += (now - last) / 1000;
      last = now;
      while (this.next < rec.messages.length && rec.messages[this.next].t * scale <= this.clock) {
        const m = rec.messages[this.next++];
        if (m.e) this.onEvent(m.e);
        else if (m.f) this.onFrame({ name: m.f.name, shape: m.f.shape, step: m.f.step, data: float16ToFloat32(m.f.b64) });
      }
      if (this.next >= rec.messages.length) this.close();
    }, 40);
  }
  control(action: string) {
    if (action === 'pause') { this.paused = true; this.onEvent({ type: 'status', state: 'paused' }); }
    if (action === 'resume') { this.paused = false; this.onEvent({ type: 'status', state: 'running' }); }
    if (action === 'stop') { this.close(); this.onEvent({ type: 'done', stopped: true, summary: null }); }
  }
  close() { if (this.timer) clearInterval(this.timer); this.timer = null; }
}

const replays = new Map<string, Replay>();

export function demoStream(id: string, onEvent: (e: RunEvent) => void, onFrame: (f: Frame) => void): { close(): void } {
  let closed = false;
  void load<Recording>(id).then((rec) => { if (!closed) replays.set(id, new Replay(rec, onEvent, onFrame)); })
    .catch((err) => onEvent({ type: 'error', message: err.message }));
  return { close: () => { closed = true; replays.get(id)?.close(); } };
}

/** Answers the REST calls the labs make, from recorded data. */
export async function demoApi<T>(path: string, init?: RequestInit): Promise<T> {
  const index = await load<{ problems: unknown; operators: unknown }>('index');
  const body = init?.body ? JSON.parse(String(init.body)) : {};
  if (path === '/api/problems') return index.problems as T;
  if (path === '/api/operators/defaults') return index.operators as T;
  if (path === '/api/runs') {
    const id = body.kind === 'pinn' ? `pinn_${body.problem}` : body.kind === 'operator' ? `operator_${body.config.model}` : 'benchmark';
    await load(id);
    return { id } as T;
  }
  const [, id, action] = path.match(/^\/api\/runs\/([^/]+)\/(\w+)$/) ?? [];
  if (action === 'control') { replays.get(id)?.control(body.action); return { state: body.action } as T; }
  if (action === 'evaluate') {
    const inputs = (await load<Recording>(id)).new_inputs ?? [];
    if (!inputs.length) throw new Error('This recording has no new-input evaluations');
    return inputs.reduce((best, s) => (Math.abs(s.length_scale - body.length_scale) < Math.abs(best.length_scale - body.length_scale) ? s : best)) as T;
  }
  throw new Error(`Not available in demo mode: ${path}`);
}

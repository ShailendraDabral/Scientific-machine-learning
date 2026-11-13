// Typed access to the NeuralPhysics API. The front end computes nothing scientific:
// it sends configurations and renders the arrays and numbers the back end returns.

export interface ParamSpec {
  name: string; label: string; group: 'pde' | 'initial' | 'boundary'; default: number | string;
  kind: 'float' | 'int' | 'choice'; min: number | null; max: number | null; step: number | null; choices: string[];
}
export interface PinnConfig {
  hidden_layers: number; neurons: number; activation: string; learning_rate: number; epochs: number;
  lbfgs_iterations: number; n_collocation: number; n_boundary: number; n_initial: number;
  w_pde: number; w_boundary: number; w_initial: number; seed: number; dtype: string;
  log_every: number; snapshot_every: number;
}
export interface ProblemInfo {
  name: string; title: string; equation: string; coords: string[]; time_dependent: boolean;
  eval_shape: number[]; loss_terms: string[]; analytical_method: string | null; numerical_method: string;
  params: ParamSpec[]; preset: PinnConfig;
}
export interface GridInfo { names: string[]; axes: number[][] }
export interface Frame { name: string; shape: number[]; step: number; data: Float32Array }
export interface MetricsEvent {
  type: 'metrics'; step: number; phase: string; losses: Record<string, number>; lr: number; elapsed: number;
  test_rel_l2?: number; model?: string;
}
export type RunState = 'idle' | 'queued' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type RunEvent = { type: string; [key: string]: any };

import { demoApi, demoStream } from './demo';

/** True in the static demo build: recorded runs are replayed and no back end is contacted. */
export const DEMO = import.meta.env.VITE_DEMO === '1';

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (DEMO) return demoApi<T>(path, init);
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? res.statusText);
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const post = <T,>(path: string, body: unknown) => api<T>(path, { method: 'POST', body: JSON.stringify(body) });

/** Binary frame: uint32 LE header length | padded JSON header | float32 LE payload (4-byte aligned). */
export function decodeFrame(buf: ArrayBuffer): Frame {
  const n = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { name: header.name, shape: header.shape, step: header.step ?? 0, data: new Float32Array(buf, 4 + n) };
}

export function openStream(runId: string, onEvent: (e: RunEvent) => void, onFrame: (f: Frame) => void): { close(): void } {
  if (DEMO) return demoStream(runId, onEvent, onFrame);
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/ws/runs/${runId}`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (msg) => {
    if (typeof msg.data === 'string') onEvent(JSON.parse(msg.data));
    else onFrame(decodeFrame(msg.data as ArrayBuffer));
  };
  return ws;
}

export const control = (runId: string, action: 'pause' | 'resume' | 'stop') =>
  post<{ state: RunState }>(`/api/runs/${runId}/control`, { action });

export const sci = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v === 0 ? '0' : v.toExponential(digits);

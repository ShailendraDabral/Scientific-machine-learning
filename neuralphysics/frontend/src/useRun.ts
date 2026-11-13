import { useCallback, useEffect, useRef, useState } from 'react';
import { control, openStream, post, type Frame, type MetricsEvent, type RunEvent, type RunState } from './api';

export interface Snapshot { step: number; prediction?: Frame; abs_error?: Frame; rel_l2?: number; max_abs?: number }

export interface RunStore {
  id: string | null;
  setup: RunEvent | null;
  metrics: MetricsEvent[];
  snapshots: Snapshot[];        // training history, for the timeline
  frames: Record<string, Frame>; // newest frame per field name
  events: RunEvent[];           // other events (stages, per-instance results)
  summary: any;                 // eslint-disable-line @typescript-eslint/no-explicit-any
}

const empty = (): RunStore => ({ id: null, setup: null, metrics: [], snapshots: [], frames: {}, events: [], summary: null });
const MAX_SNAPSHOTS = 240;

/**
 * Lifecycle of one back-end run. Streamed data is kept in a mutable store and the
 * component re-renders at most about 8 times a second, however fast events arrive.
 */
export function useRun() {
  const [state, setState] = useState<RunState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const store = useRef<RunStore>(empty());
  const socket = useRef<{ close(): void } | null>(null);
  const dirty = useRef(false);

  useEffect(() => {
    const timer = setInterval(() => {
      if (dirty.current) { dirty.current = false; setVersion((v) => v + 1); }
    }, 120);
    return () => { clearInterval(timer); socket.current?.close(); };
  }, []);

  const snapshotAt = (step: number): Snapshot => {
    const s = store.current;
    let snap = s.snapshots[s.snapshots.length - 1];
    if (!snap || snap.step !== step) {
      snap = { step };
      s.snapshots.push(snap);
      if (s.snapshots.length > MAX_SNAPSHOTS) s.snapshots.splice(1, 1); // keep the first, drop the oldest after it
    }
    return snap;
  };

  const start = useCallback(async (body: unknown) => {
    socket.current?.close();
    store.current = empty();
    setError(null);
    try {
      const run = await post<{ id: string }>('/api/runs', body);
      store.current.id = run.id;
      setState('running');
      socket.current = openStream(run.id, (e) => {
        const s = store.current;
        if (e.type === 'setup') s.setup = e;
        else if (e.type === 'metrics') s.metrics.push(e as MetricsEvent);
        else if (e.type === 'snapshot') Object.assign(snapshotAt(e.step), { rel_l2: e.rel_l2, max_abs: e.max_abs });
        else if (e.type === 'status') setState(e.state);
        else if (e.type === 'done') { s.summary = e.summary; setState(e.stopped ? 'cancelled' : 'completed'); }
        else if (e.type === 'error') { setError(e.message); setState('failed'); }
        else if (e.type !== 'hello') s.events.push(e);
        dirty.current = true;
      }, (frame) => {
        store.current.frames[frame.name] = frame;
        if (frame.name === 'prediction' || frame.name === 'abs_error') snapshotAt(frame.step)[frame.name] = frame;
        dirty.current = true;
      });
    } catch (err) {
      setError((err as Error).message);
      setState('failed');
    }
  }, []);

  const send = useCallback(async (action: 'pause' | 'resume' | 'stop') => {
    if (store.current.id) await control(store.current.id, action).catch((err) => setError(err.message));
  }, []);

  return {
    state, error, version, store: store.current, start,
    pause: () => send('pause'), resume: () => send('resume'), stop: () => send('stop'),
  };
}

import { memo, useEffect, useMemo, useState } from 'react';
import { api, DEMO, sci, type GridInfo, type PinnConfig, type ProblemInfo } from '../api';
import { Heatmap, Plot } from '../components/Plot';
import { NetworkDiagram } from '../components/NetworkDiagram';
import { Surface3D } from '../components/Surface3D';
import { Card, KindTag, NumberField, ParamField, RunControls, SelectField, SliderField, StateTag, Tex } from '../components/Widgets';
import { range, toView } from '../fields';
import { useRun } from '../useRun';

const ACTIVATIONS = ['tanh', 'sine', 'gelu', 'silu', 'softplus', 'relu'];
const LOSS_STYLE: Record<string, { label: string; color: string }> = {
  total: { label: 'Training loss (weighted total)', color: '#e6edf3' },
  pde: { label: 'PDE residual loss', color: '#4cc2ff' },
  boundary: { label: 'Boundary loss', color: '#f2cc60' },
  initial: { label: 'Initial condition loss', color: '#7ee787' },
  initial_velocity: { label: 'Initial velocity loss', color: '#d2a8ff' },
};
const GROUPS = [
  { key: 'pde', title: 'PDE parameters' },
  { key: 'initial', title: 'Initial condition' },
  { key: 'boundary', title: 'Boundary condition' },
] as const;

const MemoHeatmap = memo(Heatmap);
const MemoSurface = memo(Surface3D);

export function PinnLab() {
  const [problems, setProblems] = useState<ProblemInfo[]>([]);
  const [name, setName] = useState('heat_1d');
  const [params, setParams] = useState<Record<string, number | string>>({});
  const [cfg, setCfg] = useState<PinnConfig | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tIndex, setTIndex] = useState(0);
  const [frameIndex, setFrameIndex] = useState<number | null>(null); // null follows the live run
  const [refChoice, setRefChoice] = useState<'analytical' | 'numerical'>('analytical');
  const [surfaceField, setSurfaceField] = useState<'prediction' | 'reference' | 'abs_error'>('prediction');
  const run = useRun();

  useEffect(() => {
    api<ProblemInfo[]>('/api/problems').then(setProblems).catch((e) => setLoadError(e.message));
  }, []);
  const problem = problems.find((p) => p.name === name);
  useEffect(() => {
    if (!problem) return;
    setParams(Object.fromEntries(problem.params.map((p) => [p.name, p.default])));
    setCfg(problem.preset);
    setTIndex(0);
  }, [problem]);

  const s = run.store;
  const active = run.state === 'running' || run.state === 'paused' || run.state === 'queued';
  const locked = active || DEMO; // demo recordings have fixed settings
  const grid = (s.setup?.grid ?? null) as GridInfo | null;
  const snaps = s.snapshots;
  const snap = snaps.length ? snaps[Math.min(frameIndex ?? snaps.length - 1, snaps.length - 1)] : undefined;
  const hasAnalytical = Boolean(s.frames.analytical);
  const refKind = hasAnalytical ? refChoice : 'numerical';
  const refFrame = s.frames[refKind];
  const pred = snap?.prediction;
  const err = snap?.abs_error;
  const nT = pred && pred.shape.length === 3 ? pred.shape[2] : 0;

  const refRange = useMemo(() => range(refFrame?.data), [refFrame]);
  const predView = useMemo(() => (pred && grid ? toView(pred.data, pred.shape, grid, tIndex) : null), [pred, grid, tIndex]);
  const refView = useMemo(() => (refFrame && grid ? toView(refFrame.data, refFrame.shape, grid, tIndex) : null), [refFrame, grid, tIndex]);
  const errView = useMemo(() => (err && grid ? toView(err.data, err.shape, grid, tIndex) : null), [err, grid, tIndex]);
  const residual = s.frames.residual;
  const resView = useMemo(() => (residual && grid ? toView(residual.data, residual.shape, grid, tIndex) : null), [residual, grid, tIndex]);
  const surfView = surfaceField === 'prediction' ? predView : surfaceField === 'reference' ? refView : errView;
  const surfRange = useMemo<[number, number]>(
    () => (surfaceField === 'abs_error' ? range(err?.data) : refRange), [surfaceField, err, refRange]);

  const last = s.metrics[s.metrics.length - 1];
  const totalSteps = cfg ? cfg.epochs + cfg.lbfgs_iterations : 0;
  const lbfgsStart = s.metrics.find((m) => m.phase === 'lbfgs')?.step;
  const lossTraces = Object.keys(last?.losses ?? {}).sort((a) => (a === 'total' ? -1 : 0)).map((term) => ({
    type: 'scatter', mode: 'lines', name: LOSS_STYLE[term]?.label ?? term,
    x: s.metrics.map((m) => m.step), y: s.metrics.map((m) => m.losses[term]),
    line: { color: LOSS_STYLE[term]?.color, width: term === 'total' ? 2.2 : 1.4 },
  }));
  const errSnaps = snaps.filter((x) => x.rel_l2 !== undefined);
  const summary = s.summary;

  const setC = <K extends keyof PinnConfig>(key: K, value: PinnConfig[K]) => setCfg((c) => (c ? { ...c, [key]: value } : c));
  const start = () => {
    if (!cfg) return;
    setFrameIndex(null);
    void run.start({ kind: 'pinn', problem: name, params, config: cfg });
  };

  if (loadError) return <p className="p-6 text-sm text-red-400">Cannot reach the NeuralPhysics API: {loadError}</p>;
  if (!problem || !cfg) return <p className="p-6 text-sm text-muted">Loading the problem catalogue…</p>;

  return (
    <div className="grid grid-cols-[320px_minmax(0,1fr)] gap-3 p-3">
      {/* ------------------------------------------------ configuration */}
      <aside className="space-y-3">
        <Card title="Experiment">
          <SelectField label="Equation" value={name} onChange={setName} disabled={active}
            options={problems.map((p) => ({ value: p.name, label: p.title }))} />
          <div className="mt-2 text-sm"><Tex tex={problem.equation} block /></div>
        </Card>
        {GROUPS.map((g) => {
          const specs = problem.params.filter((p) => p.group === g.key);
          return specs.length ? (
            <Card key={g.key} title={g.title}>
              <div className="space-y-2">
                {specs.map((spec) => (
                  <ParamField key={spec.name} spec={spec} value={params[spec.name] ?? spec.default} disabled={locked}
                    onChange={(v) => setParams((p) => ({ ...p, [spec.name]: v }))} />
                ))}
              </div>
            </Card>
          ) : null;
        })}
        <Card title="Network">
          <div className="space-y-2">
            <SliderField label="Hidden layers" value={cfg.hidden_layers} min={1} max={8} step={1} disabled={locked}
              onChange={(v) => setC('hidden_layers', v)} />
            <SliderField label="Neurons per layer" value={cfg.neurons} min={4} max={128} step={4} disabled={locked}
              onChange={(v) => setC('neurons', v)} />
            <SelectField label="Activation function" value={cfg.activation} disabled={locked}
              onChange={(v) => setC('activation', v)} options={ACTIVATIONS.map((a) => ({ value: a, label: a }))} />
          </div>
        </Card>
        <Card title="Training">
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Learning rate" value={cfg.learning_rate} min={1e-5} max={0.1} step={0.0005} disabled={locked}
              onChange={(v) => setC('learning_rate', v)} />
            <NumberField label="Epochs (Adam)" value={cfg.epochs} min={0} max={20000} step={100} disabled={locked}
              onChange={(v) => setC('epochs', Math.round(v))} />
            <NumberField label="Collocation points" value={cfg.n_collocation} min={100} max={20000} step={500} disabled={locked}
              onChange={(v) => setC('n_collocation', Math.round(v))} />
            <NumberField label="L-BFGS iterations" value={cfg.lbfgs_iterations} min={0} max={5000} step={100} disabled={locked}
              onChange={(v) => setC('lbfgs_iterations', Math.round(v))} />
          </div>
          <div className="mt-3">
            <RunControls state={run.state} onStart={start} onPause={run.pause} onResume={run.resume} onStop={run.stop} />
          </div>
          {run.error && <p className="mt-2 text-xs text-red-400">{run.error}</p>}
        </Card>
      </aside>

      {/* ------------------------------------------------ results */}
      <main className="min-w-0 space-y-3">
        <Card>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
            <StateTag state={run.state} />
            <span>step <b>{last?.step ?? 0}</b> / {totalSteps}{last ? ` · ${last.phase === 'lbfgs' ? 'L-BFGS' : 'Adam'}` : ''}</span>
            <span>training loss <b>{sci(last?.losses.total)}</b></span>
            <span>relative L2 error <b>{sci(snap?.rel_l2)}</b>{snap?.rel_l2 !== undefined && <span className="text-muted"> vs {s.setup?.reference_kind}</span>}</span>
            <span className="text-muted">{last ? `${last.elapsed.toFixed(1)} s` : DEMO ? 'Press Train to replay a recorded training run' : 'Press Train to start a real training run'}</span>
          </div>
          <div className="mt-2 h-1 overflow-hidden rounded bg-line">
            <div className="h-full bg-accent transition-all" style={{ width: `${totalSteps ? Math.min(100, (100 * (last?.step ?? 0)) / totalSteps) : 0}%` }} />
          </div>
        </Card>

        <div className="grid grid-cols-3 gap-3">
          <Card title="Predicted solution" right={<KindTag kind="learned" />}>
            <MemoHeatmap view={predView} zmin={refRange[0]} zmax={refRange[1]} />
          </Card>
          <Card title={refKind === 'analytical' ? 'Analytical solution' : 'Numerical solution'}
            right={<>
              {hasAnalytical && (
                <select className="input !w-auto !py-0 text-xs" value={refChoice} onChange={(e) => setRefChoice(e.target.value as 'analytical' | 'numerical')}>
                  <option value="analytical">analytical</option><option value="numerical">numerical</option>
                </select>)}
              <KindTag kind={refKind} />
            </>}>
            <MemoHeatmap view={refView} zmin={refRange[0]} zmax={refRange[1]} />
          </Card>
          <Card title="Absolute error map" right={<span>|learned − {s.setup?.reference_kind ?? 'reference'}|</span>}>
            <MemoHeatmap view={errView} colorscale="Inferno" />
          </Card>
        </div>

        <Card>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <label className="block">
              <span className="field-label flex justify-between">
                <span>Training timeline: step {snap?.step ?? 0}</span>
                <button className="text-accent disabled:text-muted" disabled={frameIndex === null} onClick={() => setFrameIndex(null)}>
                  {frameIndex === null ? 'following live' : 'back to live'}
                </button>
              </span>
              <input type="range" min={0} max={Math.max(0, snaps.length - 1)} value={frameIndex ?? Math.max(0, snaps.length - 1)}
                disabled={snaps.length < 2} onChange={(e) => setFrameIndex(Number(e.target.value))} />
            </label>
            {nT > 0 && grid ? (
              <label className="block">
                <span className="field-label">Time slice: t = {grid.axes[2][Math.min(tIndex, nT - 1)].toFixed(2)}</span>
                <input type="range" min={0} max={nT - 1} value={tIndex} onChange={(e) => setTIndex(Number(e.target.value))} />
              </label>
            ) : <p className="self-end text-xs text-muted">Drag the timeline to replay how the solution formed during training.</p>}
          </div>
        </Card>

        <div className="grid grid-cols-2 gap-3">
          <Card title="Loss terms" right={lbfgsStart !== undefined ? <span>dashed line: Adam → L-BFGS</span> : null}>
            <Plot height={270} data={lossTraces} layout={{
              xaxis: { title: { text: 'training step', standoff: 4 } }, yaxis: { type: 'log', exponentformat: 'e' },
              shapes: lbfgsStart !== undefined ? [{ type: 'line', x0: lbfgsStart, x1: lbfgsStart, yref: 'paper', y0: 0, y1: 1,
                line: { color: '#8b98a9', dash: 'dash', width: 1 } }] : [],
            }} />
          </Card>
          <Card title="3D view (Three.js)" right={
            <select className="input !w-auto !py-0 text-xs" value={surfaceField}
              onChange={(e) => setSurfaceField(e.target.value as typeof surfaceField)}>
              <option value="prediction">predicted</option><option value="reference">{refKind}</option><option value="abs_error">absolute error</option>
            </select>}>
            <MemoSurface view={surfView} zRange={surfRange} height={270} />
          </Card>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Card title="Neural network architecture">
            <NetworkDiagram inputs={problem.coords} hiddenLayers={cfg.hidden_layers} neurons={cfg.neurons} activation={cfg.activation} />
          </Card>
          <Card title="Prediction error during training" right={<span>relative L2 vs {s.setup?.reference_kind ?? 'reference'}</span>}>
            <Plot height={200} data={[{ type: 'scatter', mode: 'lines+markers', x: errSnaps.map((x) => x.step), y: errSnaps.map((x) => x.rel_l2),
              marker: { size: 4 }, line: { color: '#ff9bce', width: 1.6 }, name: 'relative L2 error' }]}
              layout={{ showlegend: false, xaxis: { title: { text: 'training step', standoff: 4 } }, yaxis: { type: 'log', exponentformat: 'e' } }} />
          </Card>
        </div>

        {summary?.metrics && (
          <Card title="Final measurements" right={<span>{summary.n_parameters?.toLocaleString()} parameters{summary.stopped_early ? ' · stopped early' : ''}</span>}>
            <div className="grid grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] gap-4">
              <table className="w-full text-left text-xs">
                <thead className="text-muted">
                  <tr>
                    <th className="py-1 pr-2 font-medium">Solution</th><th className="pr-2 font-medium">PDE residual (RMS)</th>
                    <th className="pr-2 font-medium">Boundary error (max)</th><th className="pr-2 font-medium">Initial error (max)</th>
                    <th className="pr-2 font-medium">Rel. L2 vs analytical</th><th className="pr-2 font-medium">Rel. L2 vs numerical</th><th className="font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {(['analytical', 'numerical', 'learned'] as const).filter((k) => summary.metrics[k]).map((k) => {
                    const m = summary.metrics[k];
                    return (
                      <tr key={k} className="border-t border-line align-top">
                        <td className="py-1.5 pr-2"><KindTag kind={k} /><div className="mt-1 text-[10px] leading-snug text-muted">{summary.solutions[k]?.method}</div></td>
                        <td className="pr-2">{sci(m.pde_residual?.rms)}</td>
                        <td className="pr-2">{sci(m.conditions?.boundary?.max)}</td>
                        <td className="pr-2">{sci(m.conditions?.initial?.max)}{m.conditions?.initial_velocity ? ` · u_t ${sci(m.conditions.initial_velocity.max)}` : ''}</td>
                        <td className="pr-2">{sci(m.error_vs_analytical?.rel_l2)}</td>
                        <td className="pr-2">{sci(m.error_vs_numerical?.rel_l2)}</td>
                        <td>{summary.solutions[k]?.seconds} s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div>
                <p className="card-title mb-1">PDE residual of the learned solution</p>
                <MemoHeatmap view={resView} colorscale="RdBu" height={200} />
              </div>
            </div>
            {!summary.metrics.analytical && <p className="mt-2 text-xs text-muted">No analytical solution exists for these settings; errors are measured against the numerical solver.</p>}
          </Card>
        )}
      </main>
    </div>
  );
}

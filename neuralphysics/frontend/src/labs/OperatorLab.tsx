import { memo, useEffect, useMemo, useState } from 'react';
import { api, DEMO, post, sci, type Frame, type GridInfo } from '../api';
import { Heatmap, Plot } from '../components/Plot';
import { Card, KindTag, NumberField, RunControls, SelectField, SliderField, StateTag, Tex } from '../components/Widgets';
import { range, toView, type View2D } from '../fields';
import { useRun } from '../useRun';
import { THEORY } from './theory';

type Model = 'deeponet' | 'fno' | 'pino';
type Split = 'test' | 'ood' | 'new';
type Dict = Record<string, number | string>;
interface NewSample { u0: number[]; reference: number[][]; prediction: number[][]; abs_error: number[][]; rel_l2: number; x: number[]; t: number[] }

const MODEL_COLOR: Record<string, string> = { deeponet: '#f2cc60', fno: '#4cc2ff', pino: '#d2a8ff', pinn: '#ff9bce' };
const MODEL_LABEL: Record<string, string> = { deeponet: 'DeepONet', fno: 'FNO', pino: 'PINO', pinn: 'PINN' };
const LOSS_LABEL: Record<string, string> = { total: 'training loss', data: 'data loss', pde: 'PDE residual loss', initial: 'initial condition loss' };
const MemoHeatmap = memo(Heatmap);

/** One sample out of a stacked (n, ...) field. */
function sample(frame: Frame | undefined, i: number): { data: Float32Array; shape: number[] } | null {
  if (!frame) return null;
  const shape = frame.shape.slice(1);
  const size = shape.reduce((a, b) => a * b, 1);
  return { data: frame.data.subarray(i * size, (i + 1) * size), shape };
}
const rowsView = (rows: number[][], x: number[], t: number[]): View2D =>
  ({ z: rows.map((r) => Float32Array.from(r)), x: t, y: x, xLabel: 't', yLabel: 'x' });

export function OperatorLab() {
  const [model, setModel] = useState<Model>('fno');
  const [data, setData] = useState<Dict | null>(null);
  const [cfg, setCfg] = useState<Dict | null>(null);
  const [bench, setBench] = useState<Dict | null>(null);
  const [recorded, setRecorded] = useState<Record<string, Dict> | null>(null); // demo: settings of each recording
  const [split, setSplit] = useState<Split>('test');
  const [index, setIndex] = useState(0);
  const [newScale, setNewScale] = useState(0.08);
  const [newSeed, setNewSeed] = useState(1);
  const [newSample, setNewSample] = useState<NewSample | null>(null);
  const [benchInstance, setBenchInstance] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  const run = useRun();
  const cmp = useRun();

  useEffect(() => {
    api<{ data: Dict; config: Dict; benchmark: Dict; configs?: Record<string, Dict> }>('/api/operators/defaults')
      .then((d) => { setData(d.data); setCfg(d.configs?.fno ?? d.config); setBench(d.benchmark); setRecorded(d.configs ?? null); })
      .catch((e) => setMsg(e.message));
  }, []);

  const s = run.store;
  const grid = (s.setup?.grid ?? null) as GridInfo | null;
  const active = run.state === 'running' || run.state === 'paused' || run.state === 'queued';
  const locked = active || DEMO;
  const trained = Boolean(s.frames.pred_test);
  const monitor = (s.setup?.monitor_index ?? 0) as number;
  const metrics = s.summary?.metrics;
  const theory = THEORY[model];

  // ---- which sample is on screen -------------------------------------------------------
  const live = !trained;
  const shown = live ? monitor : index;
  const srcSplit = live || split === 'new' ? 'test' : split;
  const u0 = sample(s.frames[`u0_${srcSplit}`], shown);
  const ref = sample(s.frames[`u_${srcSplit}`], shown);
  const pred = live ? s.frames.prediction : sample(s.frames[`pred_${srcSplit}`], shown);
  const err = live ? s.frames.abs_error : sample(s.frames[`abserr_${srcSplit}`], shown);
  const useNew = trained && split === 'new' && newSample;
  const refView = useMemo(() => (useNew ? rowsView(newSample.reference, newSample.x, newSample.t)
    : ref && grid ? toView(ref.data, ref.shape, grid) : null), [useNew, newSample, ref?.data, grid]); // eslint-disable-line react-hooks/exhaustive-deps
  const predView = useMemo(() => (useNew ? rowsView(newSample.prediction, newSample.x, newSample.t)
    : pred && grid ? toView(pred.data, pred.shape, grid) : null), [useNew, newSample, pred?.data, grid]); // eslint-disable-line react-hooks/exhaustive-deps
  const errView = useMemo(() => (useNew ? rowsView(newSample.abs_error, newSample.x, newSample.t)
    : err && grid ? toView(err.data, err.shape, grid) : null), [useNew, newSample, err?.data, grid]); // eslint-disable-line react-hooks/exhaustive-deps
  const zr = useMemo(() => (useNew ? range(Float32Array.from(newSample.reference.flat())) : range(ref?.data)), [useNew, newSample, ref?.data]);
  const u0Line = useNew ? { x: newSample.x, y: newSample.u0 } : u0 && grid ? { x: grid.axes[0], y: Array.from(u0.data) } : null;
  const shownErr = useNew ? newSample.rel_l2 : trained && split !== 'new' ? metrics?.[split]?.per_sample_rel_l2?.[index] : undefined;
  const nSamples = trained && split !== 'new' ? s.frames[`u0_${split}`]?.shape[0] ?? 1 : 1;

  const setD = (k: string, v: number) => setData((d) => (d ? { ...d, [k]: v } : d));
  const setC = (k: string, v: number) => setCfg((c) => (c ? { ...c, [k]: v } : c));
  const start = () => {
    setSplit('test'); setIndex(0); setNewSample(null);
    void run.start({ kind: 'operator', data, config: { ...cfg, model } });
  };
  const evaluateNew = async () => {
    if (!s.id) return;
    try {
      setNewSample(await post<NewSample>(`/api/runs/${s.id}/evaluate`, { length_scale: newScale, amplitude: data?.amplitude ?? 1, seed: newSeed }));
      setSplit('new'); setMsg(null);
    } catch (e) { setMsg((e as Error).message); }
  };

  // ---- comparison --------------------------------------------------------------------------
  const c = cmp.store;
  const cmpActive = cmp.state === 'running' || cmp.state === 'paused' || cmp.state === 'queued';
  const stage = [...c.events].reverse().find((e) => e.type === 'stage');
  const pinnDone = c.events.filter((e) => e.type === 'pinn_instance').length;
  const result = c.summary;
  const benchGrid = (result?.grid ?? null) as GridInfo | null;
  const k = (c.frames.bench_u0?.shape[0] ?? 1) as number;
  const benchRef = sample(c.frames.bench_reference, Math.min(benchInstance, k - 1));
  const benchRange = useMemo(() => range(benchRef?.data), [benchRef?.data]); // eslint-disable-line react-hooks/exhaustive-deps
  const errMax = Math.max(...['deeponet', 'fno', 'pino', 'pinn'].map((m) => range(sample(c.frames[`bench_err_${m}`], Math.min(benchInstance, k - 1))?.data)[1]));
  const cmpCurves = ['deeponet', 'fno', 'pino'].map((m) => {
    const ms = c.metrics.filter((e) => e.model === m);
    return { type: 'scatter', mode: 'lines', name: MODEL_LABEL[m], x: ms.map((e) => e.step), y: ms.map((e) => e.test_rel_l2), line: { color: MODEL_COLOR[m], width: 1.6 } };
  });

  if (!data || !cfg || !bench) return <p className="p-6 text-sm text-muted">{msg ? `Cannot reach the NeuralPhysics API: ${msg}` : 'Loading…'}</p>;

  return (
    <div className="grid grid-cols-[320px_minmax(0,1fr)] gap-3 p-3">
      <aside className="space-y-3">
        <Card title="Operator model">
          <div className="grid grid-cols-3 gap-1">
            {(['deeponet', 'fno', 'pino'] as const).map((m) => (
              <button key={m} disabled={active} className={`btn !px-1 ${model === m ? 'btn-primary' : ''}`}
                onClick={() => { setModel(m); if (recorded) setCfg(recorded[m]); else setC('epochs', m === 'deeponet' ? 300 : 40); }}>{MODEL_LABEL[m]}</button>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted">{theory.tagline}</p>
        </Card>
        <Card title="Dataset (shared by every model)">
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Training pairs" value={Number(data.n_train)} min={16} max={2000} step={32} disabled={locked} onChange={(v) => setD('n_train', Math.round(v))} />
              <NumberField label="Dataset seed" value={Number(data.seed)} min={0} step={1} disabled={locked} onChange={(v) => setD('seed', Math.round(v))} />
            </div>
            <SliderField label="Viscosity ν" value={Number(data.nu)} min={0.01} max={0.2} step={0.005} disabled={locked} onChange={(v) => setD('nu', v)} />
            <SliderField label="Input length scale (train and test)" value={Number(data.length_scale)} min={0.04} max={0.3} step={0.01} disabled={locked} onChange={(v) => setD('length_scale', v)} />
            <SliderField label="Input length scale (out of distribution)" value={Number(data.ood_length_scale)} min={0.04} max={0.3} step={0.01} disabled={locked} onChange={(v) => setD('ood_length_scale', v)} />
            <p className="text-xs text-muted">Inputs are Gaussian random fields on {data.nx} points; outputs are reference solutions on a {data.nx} × {data.nt} grid.</p>
          </div>
        </Card>
        <Card title="Training">
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Epochs" value={Number(cfg.epochs)} min={1} max={1000} step={10} disabled={locked} onChange={(v) => setC('epochs', Math.round(v))} />
            <NumberField label="Learning rate" value={Number(cfg.learning_rate)} min={1e-5} max={0.1} step={0.0005} disabled={locked} onChange={(v) => setC('learning_rate', v)} />
            {model === 'deeponet' ? (<>
              <NumberField label="Branch/trunk width" value={Number(cfg.branch_width)} min={16} max={256} step={16} disabled={locked} onChange={(v) => setC('branch_width', Math.round(v))} />
              <NumberField label="Basis functions p" value={Number(cfg.basis)} min={8} max={256} step={8} disabled={locked} onChange={(v) => setC('basis', Math.round(v))} />
            </>) : (<>
              <NumberField label="Channel width" value={Number(cfg.width)} min={8} max={64} step={4} disabled={locked} onChange={(v) => setC('width', Math.round(v))} />
              <NumberField label="Fourier modes (x)" value={Number(cfg.modes_x)} min={2} max={16} step={1} disabled={locked} onChange={(v) => setC('modes_x', Math.round(v))} />
            </>)}
            {model === 'pino' && (<>
              <NumberField label="PDE weight λp" value={Number(cfg.w_pde)} min={0} max={100} step={0.01} disabled={locked} onChange={(v) => setC('w_pde', v)} />
              <NumberField label="Initial weight λ0" value={Number(cfg.w_ic)} min={0} max={100} step={0.5} disabled={locked} onChange={(v) => setC('w_ic', v)} />
            </>)}
          </div>
          <div className="mt-3"><RunControls state={run.state} onStart={start} onPause={run.pause} onResume={run.resume} onStop={run.stop} startLabel={`Train ${MODEL_LABEL[model]}`} /></div>
          {(run.error || msg) && <p className="mt-2 text-xs text-red-400">{run.error ?? msg}</p>}
        </Card>
      </aside>

      <main className="min-w-0 space-y-3">
        <Card title={`Mathematical foundations: ${theory.title}`} right={<span>{theory.reference}</span>}>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm leading-relaxed">
            {theory.blocks.map((b, i) => (
              <div key={i}><p className="text-[13px] text-ink/90">{b.text}</p>{b.tex && <Tex tex={b.tex} block />}</div>
            ))}
          </div>
        </Card>

        <div className="grid grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] gap-3">
          <Card title="Operator learning process" right={<><StateTag state={run.state} /><span>epoch {s.metrics.at(-1)?.step ?? 0} / {cfg.epochs}</span></>}>
            <Plot height={240} data={[
              ...Object.keys(s.metrics.at(-1)?.losses ?? {}).filter((t) => model === 'pino' || t === 'data').map((term) => ({
                type: 'scatter', mode: 'lines', name: LOSS_LABEL[term] ?? term, x: s.metrics.map((m) => m.step), y: s.metrics.map((m) => m.losses[term]), line: { width: 1.4 } })),
              { type: 'scatter', mode: 'lines', name: 'test error (relative L2)', x: s.metrics.map((m) => m.step), y: s.metrics.map((m) => m.test_rel_l2), line: { color: '#ff9bce', width: 2.2 } },
            ]} layout={{ xaxis: { title: { text: 'epoch', standoff: 4 } }, yaxis: { type: 'log', exponentformat: 'e' } }} />
          </Card>
          <Card title="Input function" right={<span>{useNew ? `new input, length scale ${(newSample as unknown as { length_scale: number }).length_scale}` : `${srcSplit} sample ${shown}`}</span>}>
            <Plot height={240} data={u0Line ? [{ type: 'scatter', mode: 'lines', x: u0Line.x, y: u0Line.y, line: { color: '#7ee787', width: 2 }, name: 'u₀(x)' }] : []}
              layout={{ showlegend: false, xaxis: { title: { text: 'x', standoff: 4 } }, yaxis: { title: { text: 'u₀(x)', standoff: 4 } } }} />
          </Card>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Card title="Reference output" right={<KindTag kind="numerical" />}><MemoHeatmap view={refView} zmin={zr[0]} zmax={zr[1]} /></Card>
          <Card title={`${MODEL_LABEL[model]} prediction`} right={<KindTag kind="learned" />}><MemoHeatmap view={predView} zmin={zr[0]} zmax={zr[1]} /></Card>
          <Card title="Prediction error" right={<span>relative L2 {sci(shownErr)}</span>}><MemoHeatmap view={errView} colorscale="Inferno" /></Card>
        </div>

        <Card title="Generalisation across input functions"
          right={trained ? <span>physics-residual floor of the reference data: {sci(s.summary?.reference_residual_rms)}</span> : <span>available when training finishes</span>}>
          <div className="grid grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] gap-4">
            <Plot height={220} data={trained && metrics ? (['test', 'ood'] as const).map((sp) => ({
              type: 'scatter', mode: 'markers', name: sp === 'test' ? `test set (mean ${sci(metrics[sp].rel_l2_mean)})` : `out of distribution (mean ${sci(metrics[sp].rel_l2_mean)})`,
              x: metrics[sp].per_sample_rel_l2.map((_: number, i: number) => i), y: metrics[sp].per_sample_rel_l2,
              marker: { size: 5, color: sp === 'test' ? '#4cc2ff' : '#f2cc60' } })) : []}
              layout={{ xaxis: { title: { text: 'sample index', standoff: 4 } }, yaxis: { type: 'log', exponentformat: 'e', title: { text: 'relative L2 error', standoff: 4 } } }} />
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <SelectField label="Sample set" value={split} disabled={!trained} onChange={(v) => { setSplit(v); setIndex(0); }}
                  options={[{ value: 'test', label: 'test set' }, { value: 'ood', label: 'out of distribution' }, ...(newSample ? [{ value: 'new' as Split, label: 'new input' }] : [])]} />
                <SliderField label="Sample" value={Math.min(index, nSamples - 1)} min={0} max={Math.max(0, nSamples - 1)} step={1} disabled={!trained || split === 'new'} onChange={setIndex} />
              </div>
              <div className="grid grid-cols-[1fr_70px_auto] items-end gap-2">
                <SliderField label={DEMO ? 'New input: length scale (nearest recorded)' : 'New input: length scale'} value={newScale} min={0.04} max={0.3} step={0.01} disabled={!trained} onChange={setNewScale} />
                <NumberField label="Seed" value={newSeed} min={0} step={1} disabled={!trained || DEMO} onChange={(v) => setNewSeed(Math.round(v))} />
                <button className="btn" disabled={!trained} onClick={evaluateNew}>Generate and predict</button>
              </div>
              {trained && metrics && (
                <p className="text-muted">
                  {metrics.n_parameters.toLocaleString()} parameters · trained in {metrics.train_seconds.toFixed(1)} s · inference {metrics.test.inference_ms_per_sample.toFixed(2)} ms per input ·
                  PDE residual of predictions {sci(metrics.test.pde_residual_rms)} (test), {sci(metrics.ood.pde_residual_rms)} (out of distribution)
                </p>
              )}
            </div>
          </div>
        </Card>

        <Card title="Comparison with a conventional PINN: identical inputs, documented settings"
          right={<><StateTag state={cmp.state} />{cmpActive && stage && <span>{stage.label}{stage.stage === 'pinn' ? ` · instance ${Math.min(pinnDone + 1, Number(bench.n_pinn_instances))} of ${bench.n_pinn_instances}` : ''} ({stage.index + 1}/{stage.total})</span>}</>}>
          <div className="grid grid-cols-[260px_minmax(0,1fr)] gap-4">
            <div className="space-y-2">
              <p className="text-xs text-muted">Trains DeepONet, FNO and PINO on the dataset configured on the left, then trains one PINN from scratch on each of the first K test inputs. Every method is scored against the same reference solutions.</p>
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="FNO / PINO epochs" value={Number(bench.epochs)} min={1} max={500} step={10} disabled={cmpActive || DEMO} onChange={(v) => setBench({ ...bench, epochs: Math.round(v) })} />
                <NumberField label="DeepONet epochs" value={Number(bench.deeponet_epochs)} min={1} max={2000} step={50} disabled={cmpActive || DEMO} onChange={(v) => setBench({ ...bench, deeponet_epochs: Math.round(v) })} />
                <NumberField label="PINN instances K" value={Number(bench.n_pinn_instances)} min={1} max={8} step={1} disabled={cmpActive || DEMO} onChange={(v) => setBench({ ...bench, n_pinn_instances: Math.round(v) })} />
                <NumberField label="PINN epochs (Adam)" value={Number(bench.pinn_epochs)} min={100} max={10000} step={100} disabled={cmpActive || DEMO} onChange={(v) => setBench({ ...bench, pinn_epochs: Math.round(v) })} />
                <NumberField label="PINN L-BFGS" value={Number(bench.pinn_lbfgs_iterations)} min={0} max={2000} step={50} disabled={cmpActive || DEMO} onChange={(v) => setBench({ ...bench, pinn_lbfgs_iterations: Math.round(v) })} />
              </div>
              <RunControls state={cmp.state} startLabel="Run comparison" onPause={cmp.pause} onResume={cmp.resume} onStop={cmp.stop}
                onStart={() => void cmp.start({ ...bench, kind: 'benchmark', data })} />
              {cmp.error && <p className="text-xs text-red-400">{cmp.error}</p>}
            </div>
            <div className="min-w-0">
              {!result && <Plot height={220} data={cmpCurves} layout={{ xaxis: { title: { text: 'epoch', standoff: 4 } }, yaxis: { type: 'log', exponentformat: 'e', title: { text: 'test relative L2', standoff: 4 } } }} />}
              {result && (
                <table className="w-full text-left text-xs">
                  <thead className="text-muted"><tr>
                    <th className="py-1 font-medium">Method</th><th className="pr-3 font-medium">Parameters</th><th className="pr-3 font-medium">Offline training</th>
                    <th className="pr-3 font-medium">Time per new input</th><th className="pr-3 font-medium">Rel. L2, same K inputs</th>
                    <th className="pr-3 font-medium">Rel. L2, full test set</th><th className="pr-3 font-medium">Rel. L2, out of dist.</th><th className="pr-3 font-medium">PDE residual (RMS)</th>
                  </tr></thead>
                  <tbody>
                    {result.rows.map((r: Record<string, number | string | null>) => (
                      <tr key={String(r.model)} className="border-t border-line">
                        <td className="py-1.5 font-semibold">{r.model}</td><td className="pr-3">{Number(r.n_parameters).toLocaleString()}</td>
                        <td>{r.kind === 'pinn' ? 'none' : `${Number(r.offline_training_seconds).toFixed(1)} s`}</td>
                        <td>{Number(r.seconds_per_new_input) < 1 ? `${(1000 * Number(r.seconds_per_new_input)).toFixed(2)} ms` : `${Number(r.seconds_per_new_input).toFixed(1)} s`}</td>
                        <td>{sci(r.same_instances_rel_l2 as number)}</td><td>{sci(r.test_rel_l2_mean as number | null)}</td>
                        <td>{sci(r.ood_rel_l2_mean as number | null)}</td><td>{sci(r.pde_residual_rms as number)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {result && <p className="mt-1 text-xs text-muted">PDE residual of the reference data on this grid: {sci(result.reference_residual_rms)}. The PINN has no training set, so the full-test and out-of-distribution columns do not apply to it.</p>}
            </div>
          </div>
          {result && benchGrid && (
            <div className="mt-3">
              <div className="mb-1 max-w-xs"><SliderField label="Test input shown" value={Math.min(benchInstance, k - 1)} min={0} max={k - 1} step={1} onChange={setBenchInstance} /></div>
              <div className="grid grid-cols-5 gap-2">
                <div><p className="card-title mb-1 !text-numerical">Reference</p><MemoHeatmap view={benchRef ? toView(benchRef.data, benchRef.shape, benchGrid) : null} zmin={benchRange[0]} zmax={benchRange[1]} height={150} /></div>
                {['deeponet', 'fno', 'pino', 'pinn'].map((m) => {
                  const p = sample(c.frames[`bench_${m}`], Math.min(benchInstance, k - 1));
                  return <div key={m}><p className="card-title mb-1" style={{ color: MODEL_COLOR[m] }}>{MODEL_LABEL[m]} prediction</p>
                    <MemoHeatmap view={p ? toView(p.data, p.shape, benchGrid) : null} zmin={benchRange[0]} zmax={benchRange[1]} height={150} /></div>;
                })}
                <div className="self-center text-xs text-muted">Absolute error, shared colour scale (max {sci(errMax)})</div>
                {['deeponet', 'fno', 'pino', 'pinn'].map((m) => {
                  const e = sample(c.frames[`bench_err_${m}`], Math.min(benchInstance, k - 1));
                  return <div key={m}><MemoHeatmap view={e ? toView(e.data, e.shape, benchGrid) : null} zmin={0} zmax={errMax} colorscale="Inferno" height={150} /></div>;
                })}
              </div>
              <details className="mt-2 text-xs">
                <summary className="cursor-pointer text-accent">Experimental settings (recorded with the results)</summary>
                <pre className="mt-1 max-h-72 overflow-auto rounded bg-black/40 p-2 text-[11px] leading-snug">{JSON.stringify(result.settings, null, 2)}</pre>
              </details>
            </div>
          )}
        </Card>
      </main>
    </div>
  );
}

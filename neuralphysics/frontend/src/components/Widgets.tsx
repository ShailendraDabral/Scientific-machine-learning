import katex from 'katex';
import type { ReactNode } from 'react';
import type { ParamSpec, RunState } from '../api';

export function Tex({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: block, throwOnError: false });
  return <span className={block ? 'block overflow-x-auto py-1' : ''} dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Card({ title, right, children, className = '' }:
  { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card p-3 ${className}`}>
      {(title || right) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="card-title">{title}</h3>
          <div className="flex items-center gap-2 text-xs text-muted">{right}</div>
        </div>
      )}
      {children}
    </section>
  );
}

export function NumberField({ label, value, onChange, min, max, step, disabled }: {
  label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="field-label">{label}</span>
      <input className="input" type="number" value={value} min={min} max={max} step={step ?? 'any'} disabled={disabled}
        onChange={(e) => { const v = Number(e.target.value); if (e.target.value !== '' && !Number.isNaN(v)) onChange(v); }} />
    </label>
  );
}

export function SliderField({ label, value, onChange, min, max, step, disabled }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number; disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="field-label flex justify-between"><span>{label}</span><span className="text-ink">{Number(value.toPrecision(4))}</span></span>
      <input type="range" value={value} min={min} max={max} step={step} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function SelectField<T extends string | number>({ label, value, onChange, options, disabled }: {
  label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="field-label">{label}</span>
      <select className="input" value={value} disabled={disabled}
        onChange={(e) => onChange((typeof value === 'number' ? Number(e.target.value) : e.target.value) as T)}>
        {options.map((o) => <option key={String(o.value)} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

/** One control generated from a problem's parameter schema. */
export function ParamField({ spec, value, onChange, disabled }:
  { spec: ParamSpec; value: number | string; onChange: (v: number | string) => void; disabled?: boolean }) {
  if (spec.kind === 'choice') {
    return <SelectField label={spec.label} value={String(value)} onChange={onChange} disabled={disabled}
      options={spec.choices.map((c) => ({ value: c, label: c.replace('_', ' ') }))} />;
  }
  return <SliderField label={spec.label} value={Number(value)} onChange={onChange} disabled={disabled}
    min={spec.min ?? 0} max={spec.max ?? 1} step={spec.step ?? (spec.kind === 'int' ? 1 : 0.01)} />;
}

const STATE_STYLE: Record<RunState, string> = {
  idle: 'text-muted', queued: 'text-muted', running: 'text-learned', paused: 'text-numerical',
  completed: 'text-analytical', cancelled: 'text-muted', failed: 'text-red-400',
};

export function StateTag({ state }: { state: RunState }) {
  return <span className={`tag ${STATE_STYLE[state]}`}>{state}</span>;
}

export function KindTag({ kind }: { kind: 'analytical' | 'numerical' | 'learned' }) {
  const cls = { analytical: 'text-analytical', numerical: 'text-numerical', learned: 'text-learned' }[kind];
  return <span className={`tag ${cls}`}>{kind}</span>;
}

/** Play / pause / stop controls shared by every lab. */
export function RunControls({ state, onStart, onPause, onResume, onStop, startLabel = 'Train' }: {
  state: RunState; onStart: () => void; onPause: () => void; onResume: () => void; onStop: () => void; startLabel?: string;
}) {
  const active = state === 'running' || state === 'paused' || state === 'queued';
  return (
    <div className="flex gap-2">
      {!active && <button className="btn btn-primary flex-1" onClick={onStart}>▶ {startLabel}</button>}
      {state === 'running' && <button className="btn btn-primary flex-1" onClick={onPause}>❚❚ Pause</button>}
      {state === 'paused' && <button className="btn btn-primary flex-1" onClick={onResume}>▶ Resume</button>}
      <button className="btn" disabled={!active} onClick={onStop}>■ Stop</button>
    </div>
  );
}

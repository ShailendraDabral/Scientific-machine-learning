import { useState } from 'react';
import { DEMO } from './api';
import { OperatorLab } from './labs/OperatorLab';
import { PinnLab } from './labs/PinnLab';

const TABS = [
  { id: 'pinn', label: 'PINN Training Lab', hint: 'Five equations, each solved analytically, numerically and by a physics-informed network' },
  { id: 'operator', label: 'Neural Operator Lab', hint: 'DeepONet, Fourier Neural Operator and PINO, compared with a conventional PINN' },
] as const;

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('pinn');
  const current = TABS.find((t) => t.id === tab)!;
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 flex items-center gap-6 border-b border-line bg-[#0b0f15]/95 px-4 py-2 backdrop-blur">
        <h1 className="text-base font-bold tracking-tight">Neural<span className="text-accent">Physics</span></h1>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${tab === t.id ? 'bg-panel-2 text-ink' : 'text-muted hover:text-ink'}`}>
              {t.label}
            </button>
          ))}
        </nav>
        <p className="ml-auto hidden text-xs text-muted lg:block">{current.hint}</p>
      </header>
      {DEMO && (
        <div className="border-b border-line bg-numerical/10 px-4 py-1.5 text-xs text-numerical" data-testid="demo-banner">
          <b>Demo mode.</b> You are replaying recordings of real training runs (long runs are time-compressed), so settings are fixed.
          Install the Python back end to train with your own settings; see the README.
        </div>
      )}
      {/* Both labs stay mounted so a running experiment survives a tab switch. */}
      <div className={tab === 'pinn' ? '' : 'hidden'}><PinnLab /></div>
      <div className={tab === 'operator' ? '' : 'hidden'}><OperatorLab /></div>
    </div>
  );
}

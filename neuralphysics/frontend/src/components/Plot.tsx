import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { View2D } from '../fields';

const AXIS = { gridcolor: '#1f2a38', zerolinecolor: '#2b394b', linecolor: '#2b394b', tickfont: { size: 10 } };
const BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#c3cfdd', size: 11, family: 'Inter, system-ui, sans-serif' },
  margin: { l: 50, r: 14, t: 10, b: 38 }, showlegend: true,
  legend: { orientation: 'h', y: 1.14, x: 0, font: { size: 10 } },
};

type Layout = Record<string, unknown> & { xaxis?: Record<string, unknown>; yaxis?: Record<string, unknown> };

/** Thin wrapper over Plotly.react: redraws in place whenever its props change. */
export function Plot({ data, layout = {}, height = 250 }: { data: unknown[]; layout?: Layout; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    void Plotly.react(ref.current, data, {
      ...BASE, ...layout, height,
      xaxis: { ...AXIS, ...(layout.xaxis ?? {}) }, yaxis: { ...AXIS, ...(layout.yaxis ?? {}) },
    }, { displayModeBar: false, responsive: true });
  });
  useEffect(() => {
    const el = ref.current;
    return () => { if (el) Plotly.purge(el); };
  }, []);
  return <div ref={ref} style={{ height }} />;
}

export function Heatmap({ view, zmin, zmax, colorscale = 'Viridis', height = 230 }:
  { view: View2D | null; zmin?: number; zmax?: number; colorscale?: string; height?: number }) {
  if (!view) return <div className="flex items-center justify-center text-xs text-muted" style={{ height }}>waiting for data</div>;
  return (
    <Plot height={height} data={[{
      type: 'heatmap', z: view.z, x: view.x, y: view.y, zmin, zmax, colorscale, zsmooth: 'best',
      colorbar: { thickness: 8, len: 0.9, tickfont: { size: 9 }, outlinewidth: 0 },
    }]} layout={{
      showlegend: false, margin: { l: 40, r: 6, t: 6, b: 34 },
      xaxis: { title: { text: view.xLabel, standoff: 4 } }, yaxis: { title: { text: view.yLabel, standoff: 4 } },
    }} />
  );
}

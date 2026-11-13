// Schematic of the fully connected network being trained. Layers wider than MAX_SHOWN
// are drawn with an ellipsis; the true width is printed under each layer.
const MAX_SHOWN = 8;

export function NetworkDiagram({ inputs, hiddenLayers, neurons, activation }:
  { inputs: string[]; hiddenLayers: number; neurons: number; activation: string }) {
  const sizes = [inputs.length, ...Array(hiddenLayers).fill(neurons), 1];
  const W = 520, H = 190, padX = 34, top = 14, bottom = 150;
  const xs = sizes.map((_, l) => padX + (l * (W - 2 * padX)) / (sizes.length - 1));
  const layers = sizes.map((n) => {
    const shown = Math.min(n, MAX_SHOWN);
    const gap = Math.min(18, (bottom - top) / Math.max(shown, 1));
    const y0 = (top + bottom) / 2 - ((shown - 1) * gap) / 2;
    return { n, ys: Array.from({ length: shown }, (_, i) => y0 + i * gap), truncated: n > MAX_SHOWN };
  });
  const params = sizes.slice(0, -1).reduce((sum, n, l) => sum + (n + 1) * sizes[l + 1], 0);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label={`Network with ${hiddenLayers} hidden layers of ${neurons} neurons`}>
        {layers.slice(0, -1).map((layer, l) => layer.ys.map((y1, i) => layers[l + 1].ys.map((y2, j) => (
          <line key={`${l}-${i}-${j}`} x1={xs[l]} y1={y1} x2={xs[l + 1]} y2={y2} stroke="#4cc2ff" strokeOpacity={0.13} strokeWidth={0.7} />
        ))))}
        {layers.map((layer, l) => (
          <g key={l}>
            {layer.ys.map((y, i) => (
              <circle key={i} cx={xs[l]} cy={y} r={5}
                fill={l === 0 ? '#7ee787' : l === layers.length - 1 ? '#f2cc60' : '#4cc2ff'}
                opacity={layer.truncated && i === MAX_SHOWN - 2 ? 0.25 : 1} />
            ))}
            {layer.truncated && <text x={xs[l] + 9} y={layer.ys[MAX_SHOWN - 2] + 3} fontSize={9} fill="#8b98a9">⋮</text>}
            <text x={xs[l]} y={H - 22} textAnchor="middle" fontSize={10} fill="#c3cfdd">
              {l === 0 ? inputs.join(', ') : l === layers.length - 1 ? 'u' : layer.n}
            </text>
            <text x={xs[l]} y={H - 9} textAnchor="middle" fontSize={8.5} fill="#8b98a9">
              {l === 0 ? 'input' : l === layers.length - 1 ? 'output' : activation}
            </text>
          </g>
        ))}
      </svg>
      <p className="text-xs text-muted">
        {sizes.join(' → ')} · {params.toLocaleString()} trainable parameters
      </p>
    </div>
  );
}

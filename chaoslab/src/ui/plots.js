import Plotly from 'plotly.js-dist-min';

export const PLOT_COLORS = { amber: '#E8A33D', cyan: '#4CC9F0', rose: '#F06292', green: '#7BD88F', grey: '#5E6B7A' };

const axis = (title) => ({
  title: { text: title, font: { size: 11 }, standoff: 6 },
  gridcolor: '#1B222C', zerolinecolor: '#2A3340', linecolor: '#2A3340', tickcolor: '#2A3340',
  ticks: 'outside', ticklen: 3, tickfont: { size: 10 }, automargin: true,
});

export function baseLayout({ title, x, y, extra = {} }) {
  return {
    title: { text: title, font: { size: 12, color: '#C9D1DB' }, x: 0.01, xanchor: 'left', y: 1, yanchor: 'top', pad: { t: 7, l: 4 } },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '"IBM Plex Mono", ui-monospace, monospace', color: '#8B97A6', size: 10 },
    margin: { l: 46, r: 10, t: 46, b: 34 },
    xaxis: axis(x), yaxis: axis(y),
    showlegend: true,
    legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.0, yanchor: 'bottom', font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
    uirevision: 'keep',
    ...extra,
  };
}

const config = { responsive: true, displaylogo: false, modeBarButtons: [['zoom2d', 'pan2d', 'resetScale2d', 'toImage']] };

/** A Plotly figure bound to a div; update(traces) redraws from data. */
export class Plot {
  constructor(div, layout) {
    this.div = div;
    this.layout = layout;
    Plotly.newPlot(div, [], layout, config);
  }
  update(traces, layoutPatch) {
    if (layoutPatch) Object.assign(this.layout, layoutPatch);
    Plotly.react(this.div, traces, this.layout, config);
  }
  relayout(patch) { Plotly.relayout(this.div, patch); }
  on(event, fn) { this.div.on(event, fn); }
  dispose() { Plotly.purge(this.div); }
}

export const line = (x, y, name, color, extra = {}) => ({
  x, y, name, type: 'scatter', mode: 'lines', line: { color, width: 1.2 }, hoverinfo: 'skip', ...extra,
});
export const dots = (x, y, name, color, size = 3, extra = {}) => ({
  x, y, name, type: 'scatter', mode: 'markers', marker: { color, size }, hoverinfo: 'skip', ...extra,
});

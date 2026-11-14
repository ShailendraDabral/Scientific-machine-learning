import '@fontsource/ibm-plex-sans/latin-400.css';
import '@fontsource/ibm-plex-sans/greek-400.css';
import '@fontsource/ibm-plex-sans/latin-500.css';
import '@fontsource/ibm-plex-sans/greek-500.css';
import '@fontsource/ibm-plex-sans/latin-600.css';
import '@fontsource/ibm-plex-sans/greek-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import 'katex/dist/katex.min.css';
import './styles.css';

import { el } from './ui/dom.js';
import { EngineClient } from './engine/client.js';
import { OdeLab } from './ui/odeLab.js';
import { LogisticLab } from './ui/logisticLab.js';
import lorenz from './experiments/lorenz.js';
import rossler from './experiments/rossler.js';
import duffing from './experiments/duffing.js';
import doublePendulum from './experiments/doublePendulum.js';
import threeBody from './experiments/threeBody.js';
import logistic from './experiments/logistic.js';
import analysis from './experiments/analysis.js';
import { AnalysisLab } from './ui/analysisLab.js';

const experiments = [lorenz, rossler, duffing, doublePendulum, threeBody, logistic, analysis];
const regions = Object.fromEntries(['status', 'sidebar', 'viewport', 'plots', 'transport', 'inspector']
  .map((id) => [id, document.getElementById(id)]));
const client = new EngineClient();
let current = null;

const ICONS = {
  lorenz: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C8 2 2 7 2 12c0 4 3 8 7 8 2 0 3-2 3-4 0 2 1 4 3 4 4 0 7-4 7-8 0-5-6-10-10-10z"/></svg>',
  rossler: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A7 7 0 1 0 8 12a4 4 0 1 0 4 4"/></svg>',
  duffing: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12c3-8 5-8 8 0s5 8 8 0 4-5 4-5"/></svg>',
  doublePendulum: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="4" r="2"/><line x1="12" y1="6" x2="8" y2="13"/><circle cx="8" cy="13" r="2"/><line x1="8" y1="15" x2="16" y2="20"/><circle cx="16" cy="20" r="2.5"/></svg>',
  threeBody: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="3"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2"/></svg>',
  logistic: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></svg>',
  analysis: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><circle cx="11" cy="11" r="3"/></svg>',
};

function buildSidebar() {
  const groups = new Map();
  for (const e of experiments) {
    if (!groups.has(e.group)) groups.set(e.group, []);
    groups.get(e.group).push(e);
  }
  regions.sidebar.append(
    el('div', { class: 'brand' },
      el('span', { class: 'glass-icon brand-icon', html: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V17a1 1 0 0 1-2 0v-.07A8 8 0 0 1 4.07 10H4a1 1 0 0 1 0-2h.07A8 8 0 0 1 11 4.07V4a1 1 0 0 1 2 0v.07A8 8 0 0 1 19.93 11H20a1 1 0 0 1 0 2h-.07A8 8 0 0 1 13 16.93z"/></svg>' }),
      el('div', { class: 'brand-text' },
        el('strong', {}, 'ChaosLab'),
        el('span', {}, 'nonlinear dynamics laboratory')
      )
    ),
    ...[...groups].map(([name, list]) => el('div', { class: 'nav-group' },
      el('h4', {}, name),
      list.map((e) => el('a', { href: `#${e.id}`, class: 'nav-item', 'data-id': e.id },
        el('span', { class: 'glass-icon nav-icon', html: ICONS[e.id] || '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>' }),
        el('span', { class: 'nav-title' }, e.title)
      )))),
    el('p', { class: 'nav-foot' }, 'Space pause · R reset · . step'));
}

function select(id) {
  const exp = experiments.find((e) => e.id === id) || experiments[0];
  current?.destroy();
  for (const k of ['status', 'viewport', 'plots', 'transport', 'inspector']) regions[k].replaceChildren();
  for (const a of regions.sidebar.querySelectorAll('.nav-item')) {
    a.classList.toggle('active', a.dataset.id === exp.id);
    if (a.dataset.id === exp.id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  document.title = `${exp.title} · ChaosLab`;
  current = exp.kind === 'map' ? new LogisticLab(exp, regions)
    : exp.kind === 'analysis' ? new AnalysisLab(exp, regions) : new OdeLab(exp, regions, client);
  window.chaoslab = current; // handle for inspection and automated tests
}

buildSidebar();
window.addEventListener('hashchange', () => select(location.hash.slice(1)));
select(location.hash.slice(1));

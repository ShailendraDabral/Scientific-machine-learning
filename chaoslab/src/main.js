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

function buildSidebar() {
  const groups = new Map();
  for (const e of experiments) {
    if (!groups.has(e.group)) groups.set(e.group, []);
    groups.get(e.group).push(e);
  }
  regions.sidebar.append(
    el('div', { class: 'brand' }, el('strong', {}, 'ChaosLab'), el('span', {}, 'nonlinear dynamics laboratory')),
    ...[...groups].map(([name, list]) => el('div', { class: 'nav-group' },
      el('h4', {}, name),
      list.map((e) => el('a', { href: `#${e.id}`, class: 'nav-item', 'data-id': e.id }, e.title)))),
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

import { el, inputValue, tex } from './dom.js';

/**
 * Parameter control: symbol, slider and exact numeric field, kept in sync.
 * spec: { key, tex | label, value, min, max, step, log, slider = true, unit }
 * With log: true the slider moves in log10 of the value.
 */
export function numberControl(spec, onChange) {
  const { min, max, step = 0.01, log = false } = spec;
  let value = spec.value;
  const toSlider = (v) => (log ? Math.log10(v) : v);
  const fromSlider = (s) => (log ? Number((10 ** s).toPrecision(3)) : s);

  const label = el('label', { class: 'ctl-label', html: spec.tex ? tex(spec.tex) : spec.label });
  const number = el('input', { type: 'text', inputmode: 'decimal', class: 'ctl-number', value: inputValue(value),
    'aria-label': spec.aria || spec.key });
  let slider = null;
  if (spec.slider !== false) {
    slider = el('input', {
      type: 'range', class: 'ctl-slider', 'aria-label': `${spec.aria || spec.key} slider`,
      min: toSlider(min), max: toSlider(max), step: log ? 0.01 : step, value: toSlider(value),
    });
    slider.addEventListener('input', () => {
      value = fromSlider(parseFloat(slider.value));
      number.value = inputValue(value);
      number.classList.remove('invalid');
      onChange(value);
    });
  }
  number.addEventListener('change', () => {
    const v = parseFloat(number.value);
    if (!Number.isFinite(v)) { number.classList.add('invalid'); return; }
    number.classList.remove('invalid');
    value = v;
    if (slider) slider.value = toSlider(Math.min(max, Math.max(min, v)));
    onChange(value); // out-of-range values are passed through: the core validates them
  });
  const root = el('div', { class: `ctl${slider ? '' : ' ctl-compact'}` }, label, slider, number,
    spec.unit ? el('span', { class: 'ctl-unit' }, spec.unit) : null);
  return {
    root,
    get value() { return value; },
    set(v) {
      value = v;
      number.value = inputValue(v);
      number.classList.remove('invalid');
      if (slider) slider.value = toSlider(Math.min(max, Math.max(min, v)));
    },
  };
}

export function selectControl(label, options, value, onChange) {
  const select = el('select', { class: 'ctl-select', 'aria-label': label },
    options.map(([v, text]) => el('option', { value: v, selected: v === value }, text)));
  select.addEventListener('change', () => onChange(select.value));
  return {
    root: el('div', { class: 'ctl ctl-row' }, el('label', { class: 'ctl-label' }, label), select),
    set(v) { select.value = v; },
  };
}

export function button(text, title, onClick, cls = '') {
  return el('button', { class: `btn ${cls}`, title, type: 'button', onclick: onClick }, text);
}

/** Readout table: update(rows) with rows = [[label, value], ...]. */
export function readouts() {
  const root = el('dl', { class: 'readouts' });
  let cells = [];
  return {
    root,
    update(rows) {
      if (cells.length !== rows.length) {
        root.replaceChildren();
        cells = rows.map(([label]) => {
          const dd = el('dd');
          root.append(el('dt', { html: label }), dd);
          return dd;
        });
      }
      rows.forEach(([, value, cls], i) => {
        if (cells[i].textContent !== value) cells[i].textContent = value;
        cells[i].className = cls || '';
      });
    },
  };
}

/** Status bar with named fields. */
export function statusBar(root, fields) {
  root.replaceChildren();
  const map = {};
  for (const [key, label] of fields) {
    const v = el('span', { class: 'st-value' });
    root.append(el('div', { class: `st-field st-${key}` }, label ? el('span', { class: 'st-label' }, label) : null, v));
    map[key] = v;
  }
  return {
    set(key, text, cls = '') {
      const n = map[key];
      if (n.textContent !== text) n.textContent = text;
      n.className = `st-value ${cls}`;
    },
  };
}

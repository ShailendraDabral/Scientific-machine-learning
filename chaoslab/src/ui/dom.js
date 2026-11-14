import katex from 'katex';

/** Tiny DOM builder: el('div', { class: 'x', onclick }, child, ...). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Format a number for a readout: fixed significant digits, exponent when needed. */
export function fmt(v, sig = 5) {
  if (v == null || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const s = (a >= 1e5 || a < 1e-3) ? v.toExponential(Math.max(0, sig - 2)) : v.toPrecision(sig);
  return s.replace('-', '−').replace(/e\+?/, 'e');
}

/** Compact value for an input field (round-trips through parseFloat). */
export function inputValue(v) {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-4)) return v.toExponential(4).replace(/\.?0+e/, 'e');
  return String(Number(v.toPrecision(7)));
}

export function tex(src, display = false) {
  return katex.renderToString(src, { displayMode: display, throwOnError: false });
}

export function section(title, ...children) {
  return el('section', { class: 'panel' }, el('h3', {}, title), ...children);
}

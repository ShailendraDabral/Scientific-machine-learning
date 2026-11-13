// Embed the recorded runs in dist-demo/index.html so the demo is one file that opens by double-click.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';

const dir = new URL('../public/demo/', import.meta.url);
const out = new URL('../dist-demo/index.html', import.meta.url);
const tags = readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => {
  const json = readFileSync(new URL(f, dir), 'utf8').replace(/</g, '\\u003c');
  return `<script type="application/json" id="demo-data-${f.replace('.json', '')}">${json}</script>`;
});
if (!tags.length) throw new Error('No recordings in public/demo: run `python scripts/record_demo.py` first');
const html = readFileSync(out, 'utf8');
const at = html.lastIndexOf('</body>');
writeFileSync(out, html.slice(0, at) + tags.join('\n') + html.slice(at));
console.log(`Embedded ${tags.length} recordings; dist-demo/index.html is ${(readFileSync(out).length / 1e6).toFixed(1)} MB`);

import { Engine } from './engine.js';

const engine = new Engine();
self.onmessage = (e) => {
  const res = engine.handle(e.data.msg);
  const transfer = [];
  for (const k of ['times', 'data', 'states']) if (res[k]) transfer.push(res[k].buffer);
  self.postMessage({ id: e.data.id, res }, transfer);
};

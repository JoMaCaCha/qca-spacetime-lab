import init, { QuantumSpacetimeUniverse as Universe } from '../pkg/qca_spacetime_lab.js';

let wasmReady = null;
let wasmMod = null;
let labU = null;

function ensureWasm() {
  if (!wasmReady) {
    wasmReady = init().then((mod) => {
      wasmMod = mod;
      labU = new Universe(8, 8, 1, 0.5, 0, 1);
      return mod;
    }).catch((err) => {
      wasmReady = null;
      throw err;
    });
  }
  return wasmReady;
}

function isAllowedMethod(u, name) {
  return (
    typeof name === 'string' &&
    name.length > 0 &&
    !name.startsWith('_') &&
    name !== 'constructor' &&
    name !== 'free' &&
    typeof u[name] === 'function'
  );
}

self.addEventListener('message', async (ev) => {
  const { id, method, args = [], opts = {} } = ev.data || {};
  let tempU = null;
  try {
    await ensureWasm();
    const u = Array.isArray(opts.grid)
      ? (tempU = new Universe(...opts.grid))
      : labU;

    if (Array.isArray(opts.setup)) {
      for (const [mName, mArgs] of opts.setup) {
        if (!isAllowedMethod(u, mName)) {
          throw new Error(`Disallowed worker setup method: ${String(mName)}`);
        }
        u[mName](...(Array.isArray(mArgs) ? mArgs : []));
      }
    }

    if (!isAllowedMethod(u, method)) {
      throw new Error(`Unknown or disallowed worker method: ${String(method)}`);
    }

    const res = u[method](...args);
    let out;
    if (opts.oqOut) {
      const len = u.open_quantum_out_len();
      if (!Number.isFinite(len) || len < 0) {
        throw new Error(`Invalid open_quantum_out_len: ${len}`);
      }
      out = len > 0
        ? new Float64Array(wasmMod.memory.buffer, res, len).slice()
        : new Float64Array(0);
    } else if (Number.isInteger(opts.anOutLen) && opts.anOutLen >= 0) {
      out = opts.anOutLen > 0
        ? new Float64Array(wasmMod.memory.buffer, res, opts.anOutLen).slice()
        : new Float64Array(0);
    } else if (opts.prefixLen) {
      const { hdr, countIdx, stride } = opts.prefixLen;
      const head = new Float64Array(wasmMod.memory.buffer, res, hdr);
      const count = Math.floor(head[countIdx]);
      if (!Number.isFinite(count) || count < 0 || count > 10_000_000) {
        throw new Error(`Invalid prefixed output length: ${count}`);
      }
      const total = hdr + stride * count;
      out = new Float64Array(wasmMod.memory.buffer, res, total).slice();
    } else if (ArrayBuffer.isView(res)) {
      out = res.slice();
    } else {
      out = res;
    }

    const transfer = ArrayBuffer.isView(out) ? [out.buffer] : [];
    self.postMessage({ id, ok: true, result: out }, transfer);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: String(err && err.message ? err.message : err),
    });
  } finally {
    if (tempU) {
      try { tempU.free(); } catch (_) { /* ignore */ }
    } else if (labU && typeof labU.reset_lab_buffers === 'function') {
      try { labU.reset_lab_buffers(); } catch (_) { /* ignore */ }
    }
  }
});

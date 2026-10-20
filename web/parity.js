// Parity probe for tests/test_web_parity.py: reads {config, seed, stream, actions,
// checkpoints, forks} as JSON on stdin and prints a SHA-256 of the full state at each
// checkpoint (and the fork stream ids of the final state).
"use strict";
const crypto = require("crypto");
const WF = require("./wildfire.js");

const job = JSON.parse(require("fs").readFileSync(0, "utf8"));
const cfg = { ...WF.DEFAULTS, ...job.config };

function digest(s) {
  const h = crypto.createHash("sha256");
  for (const a of [s.terrain, s.veg, s.fire, new Uint8Array(s.ash.buffer), s.wet]) h.update(a);
  h.update(`${s.t},${s.ax},${s.ay},${s.facing},${s.tank}`);
  return h.digest("hex");
}

let s = WF.initialState(cfg, job.seed, job.stream);
const out = {};
const checks = new Set(job.checkpoints);
if (checks.has(0)) out[0] = digest(s);
job.actions.forEach((a, i) => {
  s = WF.step(s, a, cfg);
  if (checks.has(i + 1)) out[i + 1] = digest(s);
});
if (job.render) { // colours of a window at 1 px per cell, integer step
  const [y0, x0, h, w] = job.render, tex = WF.textures(s.seed, s.H, s.W);
  const o = new Uint8Array(6), c = [0, 0, 0], rgb = new Uint8Array(h * w * 3);
  const L = WF.light(s.t, cfg.day_length);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const y = y0 + j, x = x0 + i, inside = y >= 0 && y < s.H && x >= 0 && x < s.W, k = inside ? y * s.W + x : 0;
    WF.cellObs(s, y, x, cfg, o);
    WF.color(o, L, s.t, inside ? tex.variant[k] : 0, inside ? tex.phase[k] : 0, c);
    rgb.set(c, (j * w + i) * 3);
  }
  out.render = Buffer.from(rgb).toString("base64");
}
if (job.forks) out.forks = WF.forkStreams(s, job.forks).map(String);
console.log(JSON.stringify(out));

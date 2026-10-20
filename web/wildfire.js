// JavaScript port of the wildfire simulator core (src/wildfire: rng, terrain, actions,
// dynamics, observe buckets, render colours) for the browser demo.
//
// The Python package is the ground truth. This port follows it operation by operation so
// that states are bit-identical for the same config, seed, stream and actions; this is
// checked by tests/test_web_parity.py (runs web/parity.js under Node).
// Single world only; forks run worlds one after another.
"use strict";

const WF = (() => {
  // ---- counter-based RNG (rng.py) -------------------------------------------------------
  const MASK = (1n << 64n) - 1n;
  const C1 = 0xff51afd7ed558ccdn;
  const C2 = 0xc4ceb9fe1a85ec53n;
  const C1H = 0xff51afd7, C1L = 0xed558ccd, C2H = 0xc4ceb9fe, C2L = 0x1a85ec53;
  const P = { SPREAD: 1, LIGHTNING: 2, GROWTH: 3, SEED: 4, DRY: 5, TERRAIN: 6, FLICKER: 7, POLICY: 8, FORK: 9 };

  function fmixBig(z) {
    z &= MASK;
    z ^= z >> 33n;
    z = (z * C1) & MASK;
    z ^= z >> 33n;
    z = (z * C2) & MASK;
    z ^= z >> 33n;
    return z;
  }

  function key(seed, stream, step, purpose) {
    let z = fmixBig(BigInt(seed) ^ 0x9e3779b97f4a7c15n);
    z = fmixBig(z ^ (BigInt(stream) & MASK));
    z = fmixBig(z ^ (BigInt(step) & MASK));
    z = fmixBig(z ^ BigInt(purpose));
    return [Number(z >> 32n), Number(z & 0xffffffffn)];
  }

  function mulhi32(a, b) {
    const a0 = a & 0xffff, a1 = a >>> 16, b0 = b & 0xffff, b1 = b >>> 16;
    const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const carry = ((p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff)) >>> 16;
    return (p11 + (p01 >>> 16) + (p10 >>> 16) + carry) >>> 0;
  }

  // fmix64 on (hi, lo) uint32 halves; result in RH, RL.
  let RH = 0, RL = 0;
  function fmix(hi, lo) {
    let h, l;
    lo = (lo ^ (hi >>> 1)) >>> 0;
    l = Math.imul(lo, C1L) >>> 0;
    h = (mulhi32(lo, C1L) + Math.imul(hi, C1L) + Math.imul(lo, C1H)) >>> 0;
    hi = h; lo = (l ^ (h >>> 1)) >>> 0;
    l = Math.imul(lo, C2L) >>> 0;
    h = (mulhi32(lo, C2L) + Math.imul(hi, C2L) + Math.imul(lo, C2H)) >>> 0;
    RH = h; RL = (l ^ (h >>> 1)) >>> 0;
  }

  const gridCache = new Map();
  function gridHash(H, W) {
    const id = H + "x" + W;
    if (!gridCache.has(id)) {
      const hi = new Uint32Array(H * W), lo = new Uint32Array(H * W);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        fmix(y, x); hi[y * W + x] = RH; lo[y * W + x] = RL;
      }
      gridCache.set(id, { hi, lo });
    }
    return gridCache.get(id);
  }

  function cellBits(k, y, x) { // bits for any coordinate (cell_hash then bits)
    fmix(y >>> 0, x >>> 0);
    fmix((RH ^ k[0]) >>> 0, (RL ^ k[1]) >>> 0);
  }
  function unit() { return (RH * 2097152 + (RL >>> 11)) * 1.1102230246251565e-16; } // 2**-53
  function uniform(seed, stream, step, purpose, y, x) {
    cellBits(key(seed, stream, step, purpose), y, x);
    return unit();
  }
  function threshold(p) {
    p = Math.min(Math.max(p, 0), 1);
    const t = BigInt(Math.trunc(Math.min(p * 18446744073709551616, 18446744073709549568)));
    return [Number(t >> 32n), Number(t & 0xffffffffn)];
  }
  const lt = (thr) => RH < thr[0] || (RH === thr[0] && RL < thr[1]);

  // ---- config ---------------------------------------------------------------------------
  const DEFAULTS = {
    height: 256, width: 256, day_length: 240, start_hour: 6.0, water_frac: 0.08, rock_frac: 0.04,
    v_max: 4, beta: 0.1, m_night: 0.5, wet_block: 128, burn_base: 3, burn_per_veg: 2,
    p_lightning: 1e-6, p_grow: 0.01, p_seed: 0.002, regrow_delay: 120, dry_rate: 2.0,
    tank: 20, infinite_water: false, window: 33, pixels_per_cell: 4, tile: 16, snapshot_every: 50,
  };
  const pyRound = (v) => { // Python round(): half to even
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  };
  const startStep = (c) => pyRound(c.start_hour / 24 * c.day_length);
  const fireMax = (c) => c.burn_base + c.burn_per_veg * c.v_max;

  // ---- terrain (terrain.py) -------------------------------------------------------------
  const GROUND = 0, WATER = 1, ROCK = 2;
  const LAKE = 0, ROCKF = 1, VEG = 2, VEG_CELL = 3, RIVER = 4, AGENT = 5;

  function valueNoise(seed, h, w, scale, octaves, field) {
    const total = new Float64Array(h * w);
    let amp = 1.0, norm = 0.0;
    for (let o = 0; o < octaves; o++) {
      const s = Math.max(scale / 2 ** o, 1.0);
      const gh = Math.ceil(h / s) + 2, gw = Math.ceil(w / s) + 2;
      const k = key(seed, 0, field * 16 + o, P.TERRAIN);
      const lat = new Float64Array(gh * gw);
      for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) { cellBits(k, y, x); lat[y * gw + x] = unit(); }
      const y0 = new Int32Array(h), ty = new Float64Array(h), x0 = new Int32Array(w), tx = new Float64Array(w);
      for (let i = 0; i < h; i++) { const f = i / s; y0[i] = Math.trunc(f); let t = f - y0[i]; ty[i] = t * t * (3 - 2 * t); }
      for (let j = 0; j < w; j++) { const f = j / s; x0[j] = Math.trunc(f); let t = f - x0[j]; tx[j] = t * t * (3 - 2 * t); }
      for (let i = 0; i < h; i++) {
        const r0 = y0[i] * gw, r1 = (y0[i] + 1) * gw, yy = ty[i];
        for (let j = 0; j < w; j++) {
          const a = lat[r0 + x0[j]], b = lat[r0 + x0[j] + 1], c = lat[r1 + x0[j]], d = lat[r1 + x0[j] + 1];
          const xx = tx[j];
          total[i * w + j] = total[i * w + j] + amp * ((a * (1 - xx) + b * xx) * (1 - yy) + (c * (1 - xx) + d * xx) * yy);
        }
      }
      norm += amp;
      amp *= 0.5;
    }
    for (let i = 0; i < total.length; i++) total[i] = total[i] / norm;
    return total;
  }

  function topK(values, allowed, k) {
    const mask = new Uint8Array(values.length);
    const idx = [];
    for (let i = 0; i < values.length; i++) if (allowed[i]) idx.push(i);
    k = Math.min(Math.max(k, 0), idx.length);
    idx.sort((a, b) => values[b] - values[a] || a - b);
    for (let i = 0; i < k; i++) mask[idx[i]] = 1;
    return mask;
  }

  function rivers(seed, h, w) {
    const k = key(seed, 0, RIVER, P.TERRAIN);
    const u = [];
    for (let i = 0; i < 16; i++) { cellBits(k, 0, i); u.push(unit()); }
    const mask = new Uint8Array(h * w);
    const n = 1 + (u[0] < 0.5 ? 1 : 0);
    for (let r = 0; r < n; r++) {
      const v = u.slice(1 + 7 * r, 8 + 7 * r);
      const vertical = v[0] < 0.5;
      const length = vertical ? h : w, across = vertical ? w : h;
      const centre = across * (0.2 + 0.6 * v[1]);
      const half = 1 + (v[6] < 0.5 ? 1 : 0);
      for (let a = 0; a < length; a++) {
        const lateral = centre + across * (
          0.08 * Math.sin(2 * Math.PI * (a / length * (1 + 2 * v[2]) + v[3]))
          + 0.03 * Math.sin(2 * Math.PI * (a / length * (4 + 4 * v[4]) + v[5])));
        const base = pyRound(lateral);
        for (let off = -half; off <= half; off++) {
          const c = Math.min(Math.max(base + off, 0), across - 1);
          if (vertical) mask[a * w + c] = 1; else mask[c * w + a] = 1;
        }
      }
    }
    return mask;
  }

  function generateTerrain(cfg, seed) {
    const h = cfg.height, w = cfg.width, n = h * w;
    const scale = Math.max(Math.min(h, w) / 6, 4);
    const river = rivers(seed, h, w);
    let nRiver = 0;
    const notRiver = new Uint8Array(n);
    for (let i = 0; i < n; i++) { nRiver += river[i]; notRiver[i] = 1 - river[i]; }
    const lakes = topK(valueNoise(seed, h, w, scale, 4, LAKE), notRiver, pyRound(cfg.water_frac * n) - nRiver);
    const notWater = new Uint8Array(n);
    for (let i = 0; i < n; i++) notWater[i] = river[i] || lakes[i] ? 0 : 1;
    const rock = topK(valueNoise(seed, h, w, scale / 2, 4, ROCKF), notWater, pyRound(cfg.rock_frac * n));
    const terrain = new Uint8Array(n);
    for (let i = 0; i < n; i++) terrain[i] = rock[i] ? ROCK : notWater[i] ? GROUND : WATER;
    const dn = valueNoise(seed, h, w, scale, 5, VEG);
    const kc = key(seed, 0, VEG_CELL, P.TERRAIN);
    const veg = new Uint8Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const d = Math.min(Math.max((dn[i] - 0.25) / 0.5, 0.0), 1.0);
      cellBits(kc, y, x);
      if (unit() < d && terrain[i] === GROUND) veg[i] = Math.min(Math.max(1 + Math.floor(d * cfg.v_max), 1), cfg.v_max);
    }
    return { terrain, veg };
  }

  function placeAgent(terrain, h, w, seed) {
    const radius = Math.max(Math.min(h, w) / 8, 2);
    const dist = (y, x) => Math.hypot(y - h / 2, x - w / 2);
    let anyNear = false, dmin = Infinity;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (terrain[y * w + x] !== GROUND) continue;
      const d = dist(y, x);
      if (d <= radius) anyNear = true;
      if (d < dmin) dmin = d;
    }
    const k = key(seed, 0, AGENT, P.TERRAIN);
    let best = Infinity, bx = 0, by = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (terrain[y * w + x] !== GROUND) continue;
      const d = dist(y, x);
      if (!(anyNear ? d <= radius : d <= dmin + 1)) continue;
      cellBits(k, y, x);
      const u = unit();
      if (u < best) { best = u; bx = x; by = y; }
    }
    return [bx, by];
  }

  function initialState(cfg, seed, stream = 0) {
    const { terrain, veg } = generateTerrain(cfg, seed);
    const [ax, ay] = placeAgent(terrain, cfg.height, cfg.width, seed);
    const n = cfg.height * cfg.width;
    return {
      H: cfg.height, W: cfg.width, terrain, veg,
      fire: new Uint8Array(n), ash: new Uint16Array(n), wet: new Uint8Array(n),
      t: startStep(cfg), seed: BigInt(seed), stream: BigInt(stream),
      ax, ay, facing: 2, tank: cfg.tank,
    };
  }

  function copyState(s) {
    return { ...s, veg: s.veg.slice(), fire: s.fire.slice(), ash: s.ash.slice(), wet: s.wet.slice() };
  }

  // ---- actions (actions.py) -------------------------------------------------------------
  const DY = [-1, 0, 1, 0], DX = [0, 1, 0, -1];
  const A = { NOOP: 0, N: 1, E: 2, S: 3, W: 4, IGNITE: 5, WATER: 6 };

  function applyAction(s, a, cfg) {
    const { H, W } = s;
    const move = a >= 1 && a <= 4;
    const facing = move ? a - 1 : s.facing;
    const ty = s.ay + DY[facing], tx = s.ax + DX[facing];
    const inside = ty >= 0 && ty < H && tx >= 0 && tx < W;
    let ax = s.ax, ay = s.ay;
    if (move && inside && s.terrain[ty * W + tx] === GROUND) { ax = tx; ay = ty; }
    const fire = s.fire.slice(), wet = s.wet.slice();
    if (a === A.IGNITE && inside) {
      const i = ty * W + tx;
      if (s.veg[i] > 0 && fire[i] === 0 && wet[i] < cfg.wet_block) fire[i] = cfg.burn_base + cfg.burn_per_veg * s.veg[i];
    }
    let tank = s.tank;
    if (a === A.WATER && (cfg.infinite_water || s.tank > 0)) {
      for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
        const by = ty + oy, bx = tx + ox;
        if (by >= 0 && by < H && bx >= 0 && bx < W) { fire[by * W + bx] = 0; wet[by * W + bx] = 255; }
      }
      if (!cfg.infinite_water) tank -= 1;
    }
    let near = false;
    for (let d = 0; d < 4; d++) {
      const ny = ay + DY[d], nx = ax + DX[d];
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && s.terrain[ny * W + nx] === WATER) near = true;
    }
    if (near || cfg.infinite_water) tank = cfg.tank;
    return { ...s, fire, wet, ax, ay, facing, tank };
  }

  // ---- dynamics (dynamics.py) -----------------------------------------------------------
  const SQ = 1 / Math.sqrt(2);
  const N8 = [[-1, 0, 1.0], [0, 1, 1.0], [1, 0, 1.0], [0, -1, 1.0], [-1, -1, SQ], [-1, 1, SQ], [1, -1, SQ], [1, 1, SQ]];

  function light(t, D) { return 0.5 * (1 - Math.cos(2 * Math.PI * ((t % D) / D))); }

  function step(state, action, cfg) {
    const s = applyAction(state, action, cfg);
    const { H, W, terrain, veg, fire, ash, wet } = s;
    const n = H * W, t = s.t;
    const L = light(t, cfg.day_length);
    const gh = gridHash(H, W), GH = gh.hi, GL = gh.lo;
    const fireNew = new Uint8Array(n), vegNew = veg.slice(), ashNew = new Uint16Array(n);
    const ignited = new Uint8Array(n);
    let anyBurning = false;
    for (let i = 0; i < n; i++) if (fire[i] > 0) { anyBurning = true; fireNew[i] = fire[i] - 1; }

    if (anyBurning) { // 3. spread
      const k = key(s.seed, s.stream, t, P.SPREAD);
      const m = cfg.m_night + (1 - cfg.m_night) * L;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (fire[i] > 0 || veg[i] === 0 || wet[i] >= cfg.wet_block) continue;
        let near = false;
        for (let d = 0; d < 8; d++) {
          const ny = y + N8[d][0], nx = x + N8[d][1];
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && fire[ny * W + nx] > 0) { near = true; break; }
        }
        if (!near) continue;
        const g = Math.max(0.0, 1 - wet[i] / cfg.wet_block);
        const q = cfg.beta * m * (veg[i] / cfg.v_max) * g;
        let keep = 1.0;
        for (let d = 0; d < 8; d++) {
          const ny = y + N8[d][0], nx = x + N8[d][1];
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && fire[ny * W + nx] > 0) keep *= 1 - q * N8[d][2];
        }
        fmix((GH[i] ^ k[0]) >>> 0, (GL[i] ^ k[1]) >>> 0);
        if (unit() < 1 - keep) ignited[i] = 1;
      }
    }
    if (cfg.p_lightning > 0) { // 4. lightning
      const k = key(s.seed, s.stream, t, P.LIGHTNING), thr = threshold(cfg.p_lightning);
      for (let i = 0; i < n; i++) {
        if (veg[i] === 0 || fire[i] > 0) continue;
        fmix((GH[i] ^ k[0]) >>> 0, (GL[i] ^ k[1]) >>> 0);
        if (lt(thr)) ignited[i] = 1;
      }
    }
    for (let i = 0; i < n; i++) {
      if (ignited[i]) fireNew[i] = cfg.burn_base + cfg.burn_per_veg * veg[i];
      const burntOut = fire[i] > 0 && fire[i] === 1;
      if (burntOut) vegNew[i] = 0;
      ashNew[i] = burntOut ? 1 : ash[i] + (ash[i] > 0 && ash[i] < 65535 ? 1 : 0);
    }
    if (L > 0) { // 5. growth and seeding
      const kg = key(s.seed, s.stream, t, P.GROWTH), ks = key(s.seed, s.stream, t, P.SEED);
      const thrG = threshold(cfg.p_grow * L);
      const thrS = [0, 1, 2, 3, 4].map((k) => threshold(cfg.p_seed * L * (1 + k)));
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, v = veg[i];
        const kk = v > 0 ? kg : ks;
        if (v > 0) {
          if (v >= cfg.v_max || fire[i] > 0 || ignited[i]) continue;
          fmix((GH[i] ^ kk[0]) >>> 0, (GL[i] ^ kk[1]) >>> 0);
          if (lt(thrG)) vegNew[i] = v + 1;
        } else {
          if (terrain[i] !== GROUND || !(ash[i] === 0 || ash[i] > cfg.regrow_delay)) continue;
          const ko = (y > 0 && veg[i - W] > 0) + (y < H - 1 && veg[i + W] > 0)
            + (x > 0 && veg[i - 1] > 0) + (x < W - 1 && veg[i + 1] > 0);
          fmix((GH[i] ^ kk[0]) >>> 0, (GL[i] ^ kk[1]) >>> 0);
          if (lt(thrS[ko])) { vegNew[i] = 1; ashNew[i] = 0; }
        }
      }
    }
    let wetNew = wet; // 6. drying
    const rate = cfg.dry_rate * (0.5 + L), whole = Math.floor(rate), frac = rate - whole;
    let kd = null;
    for (let i = 0; i < n; i++) {
      if (wet[i] === 0) continue;
      if (kd === null) { kd = key(s.seed, s.stream, t, P.DRY); wetNew = wet.slice(); }
      fmix((GH[i] ^ kd[0]) >>> 0, (GL[i] ^ kd[1]) >>> 0);
      const dec = whole + (unit() < frac ? 1 : 0);
      wetNew[i] = Math.max(wet[i] - dec, 0);
    }
    return { ...s, veg: vegNew, fire: fireNew, ash: ashNew, wet: wetNew, t: t + 1 };
  }

  function forkStreams(s, n) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const [h, l] = key(s.stream, i, s.t, P.FORK);
      out.push(((BigInt(h) << 32n) | BigInt(l)) & ((1n << 63n) - 1n));
    }
    return out;
  }

  // ---- observation buckets and colours (observe.py, render/) ----------------------------
  const fireBucket = (f, cfg) => (f > 0 ? Math.min(Math.max(Math.ceil(3 * f / fireMax(cfg)), 1), 3) : 0);
  const wetBucket = (w) => Math.floor((w + 84) / 85);
  const ashBucket = (a, cfg) => (a > 0 ? 1 + Math.min(2, Math.floor((a - 1) * 3 / Math.max(cfg.regrow_delay, 1))) : 0);

  // observable cell: [terrain, veg, burning, fire_bucket, wet_bucket, ash_bucket]
  function cellObs(s, y, x, cfg, out) {
    if (y < 0 || y >= s.H || x < 0 || x >= s.W) { out[0] = ROCK; out[1] = out[2] = out[3] = out[4] = out[5] = 0; return out; }
    const i = y * s.W + x;
    out[0] = s.terrain[i]; out[1] = s.veg[i]; out[2] = s.fire[i] > 0 ? 1 : 0;
    out[3] = fireBucket(s.fire[i], cfg); out[4] = wetBucket(s.wet[i]); out[5] = ashBucket(s.ash[i], cfg);
    return out;
  }

  // palette (render/palette.py): four base colours, everything else a rounded linear mix
  const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
  const mix = (a, b, f) => a.map((x, k) => Math.floor(x + f * (b[k] - x) + 0.5));
  const GROUND_C = hex("#0F1A17"), FIRE_C = hex("#E8833A"), INK_C = hex("#E9EFE9"), ACCENT_C = hex("#4FB3A3");
  const BARE_C = mix(GROUND_C, INK_C, 0.12), ASH_FRESH = mix(GROUND_C, INK_C, 0.24);
  const VEG_C = [BARE_C, ...[0.34, 0.48, 0.63, 0.78].map((f) => mix(GROUND_C, ACCENT_C, f))];
  const PAL = {
    ground: GROUND_C, fire: FIRE_C, ink: INK_C, accent: ACCENT_C, bare: BARE_C,
    rock: mix(GROUND_C, INK_C, 0.42), water: hex("#2A6688"), waterVariant: hex("#307596"),
    ember: mix(GROUND_C, FIRE_C, 0.62), ash: [BARE_C, ASH_FRESH, mix(ASH_FRESH, BARE_C, 0.5), BARE_C],
    veg: VEG_C, vegVariant: [BARE_C, ...VEG_C.slice(1).map((c) => mix(GROUND_C, c, 0.72))],
    wet: hex("#5FA8D3"), wetMix: 0.65, flameBase: 0.1, flameAmp: 0.1, flamePeriod: 12,
    nightMin: 0.5, gapFrac: 0.09, gapMinPx: 6, fog: GROUND_C, mix,
  };
  const VARIANT_COUNTER = 64; // TERRAIN sub-counter (render/topdown.py)

  // per-cell textures for a whole world: variant bit and flame phase (render/topdown.py)
  function textures(seed, H, W) {
    const kv = key(seed, 0, VARIANT_COUNTER, P.TERRAIN), kf = key(seed, 0, 0, P.FLICKER);
    const variant = new Uint8Array(H * W), phase = new Float64Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      cellBits(kv, y, x); variant[y * W + x] = unit() < 0.5 ? 1 : 0;
      cellBits(kf, y, x); phase[y * W + x] = unit();
    }
    return { variant, phase };
  }

  // colour of an observable cell -> out[0..2] (render/topdown.py colors)
  function color(o, L, t, variant, phase, out) {
    const [tr, v, b, fb, wb, ab] = o;
    if (b) {
      if (fb >= 2) {
        const f = PAL.flameBase + PAL.flameAmp * Math.sin(2 * Math.PI * (t / PAL.flamePeriod + phase));
        for (let j = 0; j < 3; j++) out[j] = Math.floor(FIRE_C[j] + f * (INK_C[j] - FIRE_C[j]) + 0.5);
      } else for (let j = 0; j < 3; j++) out[j] = PAL.ember[j];
      return out;
    }
    let c;
    if (tr === WATER) c = variant ? PAL.waterVariant : PAL.water;
    else if (tr === ROCK) c = PAL.rock;
    else if (v > 0) c = variant ? PAL.vegVariant[v] : PAL.veg[v];
    else c = PAL.ash[ab];
    const f = tr === GROUND ? PAL.wetMix * 85 * wb / 255 : 0;
    const k = PAL.nightMin + (1 - PAL.nightMin) * L;
    for (let j = 0; j < 3; j++) {
      const w = Math.floor(c[j] + f * (PAL.wet[j] - c[j]) + 0.5);
      out[j] = Math.floor(w * k + 0.5);
    }
    return out;
  }

  function gapPx(ppc, force = false) {
    if (ppc < PAL.gapMinPx) return force && ppc > 1 ? 1 : 0;
    return Math.max(1, Math.floor(PAL.gapFrac * ppc + 0.5));
  }

  return {
    P, key, uniform, textures, gapPx, threshold, DEFAULTS, startStep, fireMax, GROUND, WATER, ROCK,
    generateTerrain, initialState, copyState, applyAction, step, light, forkStreams,
    A, DY, DX, cellObs, color, PAL,
  };
})();

if (typeof module !== "undefined") module.exports = WF;

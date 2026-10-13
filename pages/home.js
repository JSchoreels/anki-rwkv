// Home page: hub stats and three interactive RWKV explainers (streams, delta-rule state, forgetting curve).
(() => {
  'use strict';
  const { h, bindTip, fmt, esc } = window.UI;
  const $ = (id) => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';
  const s = (tag, attrs, ...kids) => {
    const el = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    kids.flat().forEach((c) => c != null && el.append(c));
    return el;
  };

  // ---------- Hub stats ----------
  fetch('data/r01.json', { cache: 'no-cache' }).then((r) => r.json()).then((d) => {
    const x = d.headline;
    $('hub-campaign').innerHTML = `×${x.speedup_x.toFixed(2)} <small>faster · ${fmt(x.accepted_reviews_per_second)} reviews/s</small>`;
  }).catch(() => { $('hub-campaign').textContent = ''; });
  fetch('data/train-8k.json', { cache: 'no-cache' }).then((r) => r.json()).then((d) => {
    const done = (d.epochs || []).filter((e) => (e.users_done || 0) >= d.training_users).length;
    const STATUS = { training: 'training', evaluating: 'evaluating', finished: 'finished', failed: 'stopped', waiting: 'not started' };
    const label = d.mock ? 'not started' : (STATUS[d.status] || d.status);
    $('hub-train').innerHTML = `${esc(label)} <small>· ${done}/${d.epochs_planned} epochs · ${fmt(d.training_users)} users</small>`;
  }).catch(() => { $('hub-train').textContent = ''; });

  // =====================================================================
  // 1. One history, five timelines
  // =====================================================================
  const CARDS = {
    c1: { label: 'casa → house', note: 'casa', deck: 'Spanish', color: 'var(--s1)' },
    c2: { label: 'house → casa', note: 'casa', deck: 'Spanish', color: 'var(--s2)' },
    c3: { label: 'perro → dog', note: 'perro', deck: 'Spanish', color: 'var(--s3)' },
    c4: { label: '猫 → cat', note: '猫', deck: 'Japanese', color: 'var(--s4)' },
    c5: { label: 'femur', note: 'femur', deck: 'Anatomy', color: 'var(--s5)' },
  };
  const PRESET = { Spanish: 'Languages', Japanese: 'Languages', Anatomy: 'Default' };
  const G = { 1: 'A', 2: 'H', 3: 'G', 4: 'E' };
  const GNAME = { 1: 'Again', 2: 'Hard', 3: 'Good', 4: 'Easy' };
  const RAW = [
    [0, 'c1', 3], [0, 'c2', 3], [0, 'c4', 1], [0, 'c4', 3], [1, 'c1', 3], [1, 'c3', 3], [1, 'c5', 2],
    [2, 'c2', 2], [2, 'c4', 3], [3, 'c5', 3], [4, 'c1', 3], [5, 'c3', 1], [6, 'c4', 4], [8, 'c5', 3],
    [9, 'c2', 3], [10, 'c3', 3], [13, 'c1', 4],
  ];
  const seen = new Set();
  const REVIEWS = RAW.map(([day, card, grade], i) => {
    const c = CARDS[card];
    const r = { i, day, card, grade, first: !seen.has(card), note: c.note, deck: c.deck, preset: PRESET[c.deck], user: 'you' };
    seen.add(card);
    return r;
  });
  const STREAMS = [
    ['card', 'Card', (r) => r.card, (k) => CARDS[k].label],
    ['deck', 'Deck', (r) => r.deck, (k) => k],
    ['note', 'Note', (r) => r.note, (k) => `note “${k}”`],
    ['preset', 'Preset', (r) => r.preset, (k) => `preset ${k}`],
    ['user', 'User', (r) => r.user, () => 'all reviews'],
  ];
  const STREAM_NOTE = {
    card: 'The card stream links each review only to earlier reviews of the <strong>same card</strong>. This is where the classic &ldquo;interval and grade history&rdquo; lives.',
    deck: 'The deck stream links all reviews in the <strong>same deck</strong>, so it can notice that a whole deck is hard, or that you are cramming it today.',
    note: 'The note stream links <strong>sibling cards</strong>: reviewing &ldquo;casa &rarr; house&rdquo; tells the model something about &ldquo;house &rarr; casa&rdquo;.',
    preset: 'The preset stream links every card that shares <strong>deck options</strong>; here Spanish and Japanese share the Languages preset.',
    user: 'The user stream sees <strong>every review in order</strong>: your overall pace, fatigue within a session, long breaks.',
  };
  let stream = 'card', selected = 16, showQ = false;
  const pick = $('stream-pick');
  for (const [id, label] of STREAMS) {
    const b = h('button', { type: 'button', 'data-s': id, 'aria-pressed': String(id === stream) }, label);
    b.addEventListener('click', () => { stream = id; drawHist(); });
    pick.append(b);
  }
  $('show-q').addEventListener('change', (e) => { showQ = e.target.checked; drawHist(); });
  for (const [id, c] of Object.entries(CARDS)) {
    $('card-keys').append(h('span', {}, h('i', { style: { background: c.color } }), `${c.label} `, h('b', {}, c.deck)));
  }

  function drawHist() {
    [...pick.children].forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.s === stream)));
    const [, sLabel, keyOf, laneName] = STREAMS.find((x) => x[0] === stream);
    // Rows to draw: optional query copies before every non-first review.
    const rows = [];
    REVIEWS.forEach((r) => {
      if (showQ && !r.first) rows.push({ ...r, q: true });
      rows.push({ ...r, q: false });
    });
    const lanes = [...new Set(REVIEWS.map(keyOf))];
    const step = showQ ? 28 : 40, tok = showQ ? 22 : 30, left = 128, top = 26, laneH = 44;
    const W = left + rows.length * step + 20, H = top + lanes.length * laneH + 30;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, style: `min-width:${Math.min(W, 1100)}px` });
    lanes.forEach((ln, li) => {
      const y = top + li * laneH;
      svg.append(s('rect', { class: 'lane', x: 0, y: y + 2, width: W, height: laneH - 4, rx: 6 }));
      svg.append(s('text', { class: 'lanelbl', x: 8, y: y + laneH / 2 + 4 }, laneName(ln)));
    });
    // Day labels.
    let lastDay = -1;
    rows.forEach((r, i) => {
      if (r.day !== lastDay) {
        const x = left + i * step;
        svg.append(s('text', { x, y: 14 }, `day ${r.day}`));
        svg.append(s('line', { x1: x - 4, x2: x - 4, y1: 18, y2: H - 22, stroke: 'var(--line)' }));
        lastDay = r.day;
      }
    });
    const sel = REVIEWS[selected];
    const selLane = keyOf(sel);
    // Arcs: each written review links to the previous written review in the same lane; queries read only.
    const lastInLane = {};
    const pos = rows.map((r, i) => ({ x: left + i * step + tok / 2, y: top + lanes.indexOf(keyOf(r)) * laneH + laneH / 2 }));
    rows.forEach((r, i) => {
      const k = keyOf(r);
      const prev = lastInLane[k];
      if (prev != null) {
        const a = pos[prev], b = pos[i];
        const lift = Math.min(18, 6 + (b.x - a.x) / 10);
        const active = k === selLane && r.i <= sel.i;
        svg.append(s('path', {
          class: 'arc', d: `M${a.x} ${a.y - tok / 2} C${a.x} ${a.y - tok / 2 - lift} ${b.x} ${b.y - tok / 2 - lift} ${b.x} ${b.y - tok / 2}`,
          stroke: active ? 'var(--ink)' : 'var(--muted)', 'stroke-opacity': active ? 0.9 : 0.35,
          'stroke-dasharray': r.q ? '3 3' : null,
        }));
      }
      if (!r.q) lastInLane[k] = i;
    });
    rows.forEach((r, i) => {
      const c = CARDS[r.card];
      const visible = keyOf(r) === selLane && r.i <= sel.i;
      const g = s('g', { class: 'tok' + (r.q ? ' q' : '') + (visible ? '' : ' dim'), style: `--fc:${c.color}`, tabindex: 0, role: 'button' });
      g.append(s('rect', { x: pos[i].x - tok / 2, y: pos[i].y - tok / 2, width: tok, height: tok, rx: 6, fill: c.color }));
      g.append(s('text', { x: pos[i].x, y: pos[i].y + 4, 'text-anchor': 'middle' }, r.q ? '?' : G[r.grade]));
      if (r.i === sel.i && !r.q) g.append(s('rect', { x: pos[i].x - tok / 2 - 3, y: pos[i].y - tok / 2 - 3, width: tok + 6, height: tok + 6, rx: 8, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }));
      const tipText = r.q
        ? `<b>Query for review ${r.i + 1}</b>${esc(c.label)}, day ${r.day}<div class="m">Grade, answer time and review state are blanked. Predicts the grade (Instant) from memory; its own update to memory is thrown away, so later reviews never see it.</div>`
        : `<b>Review ${r.i + 1}: ${GNAME[r.grade]}</b>${esc(c.label)} · ${esc(r.deck)} · day ${r.day}${r.first ? '<div class="m">first time this card is seen</div>' : ''}`;
      bindTip(g, tipText);
      const choose = () => { selected = r.i; drawHist(); };
      g.addEventListener('click', choose);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
      svg.append(g);
    });
    $('hist').replaceChildren(svg);
    // Explanation for the selected review.
    const counts = STREAMS.map(([id, label, kf]) => {
      const n = REVIEWS.filter((r) => kf(r) === kf(sel) && r.i < sel.i).length;
      return `<strong>${label}</strong> ${n}`;
    }).join(' · ');
    $('stream-explain').innerHTML = `${STREAM_NOTE[stream]}<br>Review ${sel.i + 1} (${esc(CARDS[sel.card].label)}, day ${sel.day}) has this many earlier reviews in each stream&rsquo;s memory: ${counts}. ` +
      `The streams are stacked, so the final prediction combines all five views.` +
      (showQ ? ' <br>Dashed &ldquo;?&rdquo; squares are query rows: they use the memory to guess the grade before it is revealed, and their own update is discarded. Then the real review updates memory for good.' : '');
  }
  drawHist();

  // =====================================================================
  // 2. Delta-rule state
  // =====================================================================
  const K = 4;
  const norm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const KEYS = Array.from({ length: K }, (_, i) => norm(Array.from({ length: K }, (_, j) => (j === i ? 1 : j === (i + 1) % K ? 0.4 : 0))));
  const KCOL = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
  const KNAME = ['card A', 'card B', 'card C', 'card D'];
  // [key index, value or null for a read-only query]
  const SEQ = [
    [0, [1, 0, 0.5, 0]], [1, [0, 1, 0, -0.5]], [0, null], [2, [0.5, -0.5, 1, 0]], [0, [-1, 0.5, 0, 0.5]],
    [3, [0, 0, -0.5, 1]], [1, null], [1, [0.5, 0.5, -1, 0]], [0, null], [2, [0, 1, 0.5, 0]], [0, [0, -1, 1, 0]], [3, null],
  ];
  let S, t, last, rule = 'delta', timer = null;
  const fmt2 = (x) => (Math.abs(x) < 0.005 ? '0' : x.toFixed(2).replace('-0.', '−.').replace('0.', '.').replace('-', '−'));
  const matVec = (M, v) => M.map((row) => row.reduce((a, x, j) => a + x * v[j], 0));
  function reset() {
    S = Array.from({ length: K }, () => Array(K).fill(0));
    t = 0; last = Array(K).fill(null);
    drawWkv();
  }
  function stepWkv() {
    if (t >= SEQ.length) { stop(); return; }
    const [ki, v] = SEQ[t];
    if (v) {
      const w = +$('wkv-w').value, a = +$('wkv-a').value, k = KEYS[ki];
      const Sk = matVec(S, k);
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const erase = rule === 'delta' ? a * Sk[i] * k[j] : 0;
        S[i][j] = S[i][j] * w - erase + a * v[i] * k[j];
      }
      last[ki] = v;
      // Decay also fades what earlier writes stored, so the target the reader should get is unchanged:
      // the black ticks show the latest value written, which is what a perfect memory would return.
    }
    t++;
    drawWkv();
  }
  function stop() { clearInterval(timer); timer = null; $('wkv-play').textContent = 'Play'; }
  $('wkv-step').addEventListener('click', () => { stop(); stepWkv(); });
  $('wkv-reset').addEventListener('click', () => { stop(); reset(); });
  $('wkv-play').addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (t >= SEQ.length) reset();
    $('wkv-play').textContent = 'Pause';
    timer = setInterval(stepWkv, 900);
  });
  [...$('wkv-rule').children].forEach((b) => b.addEventListener('click', () => {
    rule = b.dataset.rule;
    [...$('wkv-rule').children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    stop(); reset();
  }));
  for (const id of ['wkv-w', 'wkv-a']) {
    $(id).addEventListener('input', () => { $(id + '-o').textContent = (+$(id).value).toFixed(2); });
    $(id).addEventListener('change', () => { stop(); reset(); });
    $(id + '-o').textContent = (+$(id).value).toFixed(2);
  }
  function cellColor(x) {
    const m = Math.min(1, Math.abs(x) / 1.2);
    if (m < 0.02) return 'var(--zero)';
    return `color-mix(in srgb, ${x > 0 ? 'var(--pos)' : 'var(--neg)'} ${Math.round(15 + 70 * m)}%, var(--zero))`;
  }
  function drawWkv() {
    // Sequence chips.
    const seq = $('wkv-seq');
    seq.replaceChildren(...SEQ.map(([ki, v], i) => {
      const chip = h('span', { class: (i === t - 1 ? 'now' : i < t - 1 ? 'done' : '') + (v ? '' : ' read'), style: { '--fc': KCOL[ki] } },
        h('i'), v ? `write ${KNAME[ki]}` : `read ${KNAME[ki]}?`);
      bindTip(chip, v ? `<b>Review of ${KNAME[ki]}</b>writes value [${v.map(fmt2).join(', ')}] at its key` : `<b>Query for ${KNAME[ki]}</b>reads S with this card\u2019s vector; skip = true, so its update is not kept`);
      return chip;
    }));
    // State heatmap.
    const st = $('wkv-state');
    st.replaceChildren(h('div', { class: 'h' }));
    for (let j = 0; j < K; j++) st.append(h('div', { class: 'h' }, `k${j + 1}`));
    for (let i = 0; i < K; i++) {
      st.append(h('div', { class: 'h' }, `v${i + 1}`));
      for (let j = 0; j < K; j++) {
        const cell = h('div', { style: { background: cellColor(S[i][j]) } }, fmt2(S[i][j]));
        st.append(cell);
      }
    }
    // Readout.
    const ro = $('wkv-read');
    ro.replaceChildren();
    for (let k = 0; k < K; k++) {
      const out = matVec(S, KEYS[k]);
      const bars = h('div', { class: 'bars' });
      out.forEach((x, i) => {
        const b = h('div', { class: x < 0 ? 'neg' : '', style: { '--v': String(Math.min(1, Math.abs(x) / 1.2)) } });
        if (last[k]) b.append(h('span', { class: 'tgt', style: { top: `calc(${50 - 50 * Math.max(-1.2, Math.min(1.2, last[k][i])) / 1.2}% - 1px)` } }));
        bindTip(b, `<b>${KNAME[k]}, dim ${i + 1}</b>memory returns ${fmt2(x)}${last[k] ? `<div class="m">last written ${fmt2(last[k][i])}</div>` : ''}`);
        bars.append(b);
      });
      const err = last[k] ? Math.sqrt(out.reduce((a, x, i) => a + (x - last[k][i]) ** 2, 0) / K) : null;
      ro.append(h('div', { class: 'ro' }, h('div', { class: 'nm', style: { '--fc': KCOL[k] } }, h('i'), KNAME[k]), bars,
        h('div', { class: 'err' }, err == null ? '–' : err.toFixed(2))));
    }
    const a = (+$('wkv-a').value).toFixed(2), w = (+$('wkv-w').value).toFixed(2);
    $('wkv-formula').textContent = rule === 'delta'
      ? `S ← S·w − a·(S·k)·kᵀ + a·v·kᵀ     (w = ${w}, a = ${a})`
      : `S ← S·w + a·v·kᵀ     (w = ${w}, a = ${a})`;
    const cur = SEQ[t - 1];
    let msg;
    if (!cur) msg = 'Press <strong>Step</strong> or <strong>Play</strong>. Card A is written three times with different values, and read back in between.';
    else if (!cur[1]) msg = `A query reads S for ${KNAME[cur[0]]} and leaves S unchanged. This is how the model guesses a grade before it sees it. (In the real model a query computes an update, uses it for its own output, then throws it away.)`;
    else if (rule === 'delta') msg = `The delta rule first <strong>erases</strong> what S already returns for this key (the a&middot;(S&middot;k)&middot;k&#7488; term), then writes the new value. With a = 1 and w = 1 the latest value for a key comes back almost exactly; the small errors come from the keys overlapping.`;
    else msg = `Without the erase term, new values are simply <strong>added</strong> on top of the old ones. Watch card A&rsquo;s error grow each time it is rewritten: the memory returns a blend of every value ever stored there.`;
    if (t >= SEQ.length) msg += ` <br>Done. Switch the rule, or lower the decay w to see older memories fade.`;
    $('wkv-explain').innerHTML = msg;
  }
  reset();

  // =====================================================================
  // 3. Forgetting-curve mixture
  // =====================================================================
  const MIN = 60, HOUR = 3600, DAY = 86400;
  const BASIS = [
    [MIN, '1m'], [10 * MIN, '10m'], [HOUR, '1h'], [6 * HOUR, '6h'], [DAY, '1d'],
    [4 * DAY, '4d'], [14 * DAY, '2w'], [60 * DAY, '2mo'], [180 * DAY, '6mo'], [730 * DAY, '2y'],
  ];
  const PRESETS = {
    'New card': [0, 2, 6, 5, 2, 1, 0.3, 0, 0, 0],
    'Mature card': [0, 0, 0, 0, 0.3, 0.6, 1.5, 5, 4, 1],
    'Leaky memory': [1.5, 3, 1, 0, 0, 0, 0.5, 2.5, 3, 0.5],
    'Single curve': [0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
  };
  let weights = PRESETS['Leaky memory'].slice(), hover = -1;
  const pre = $('curve-presets');
  for (const name of Object.keys(PRESETS)) {
    const b = h('button', { type: 'button', class: 'btn' }, name);
    b.addEventListener('click', () => { weights = PRESETS[name].slice(); drawCurve(); });
    pre.append(b);
  }
  const R = (t, sIdx) => Math.pow(0.9, t / BASIS[sIdx][0]);
  const total = () => weights.reduce((a, b) => a + b, 0) || 1;
  const basisMix = (t) => weights.reduce((a, w, i) => a + w * R(t, i), 0) / total();

  // Expertium's current GRU-style head (RWKV_GRU_HEAD=N): N curves whose weight, stability S and
  // decay d are all predicted per review. R(t) = sum_i w_i * (1 + t / S_i)^(-d_i).
  let mode = 'basis';
  const GRU = [{ w: 0.3, S: 10 * MIN, d: 0.5 }, { w: 0.4, S: 3 * DAY, d: 0.5 }, { w: 0.3, S: 180 * DAY, d: 0.5 }];
  const GCOL = ['var(--s2)', 'var(--s3)', 'var(--s7)'];
  const gruTotal = () => GRU.reduce((a, c) => a + c.w, 0) || 1;
  const gruR = (t, c) => Math.pow(1 + t / c.S, -c.d);
  const gruMix = (t) => GRU.reduce((a, c) => a + c.w * gruR(t, c), 0) / gruTotal();
  const mix = (t) => (mode === 'basis' ? basisMix(t) : gruMix(t));

  // Weight bars.
  const WB = { x0: 30, y0: 14, w: 320, h: 170 };
  const wsvg = $('curve-w');
  function drawWeights() {
    const tot = total();
    const bw = WB.w / BASIS.length;
    wsvg.replaceChildren();
    wsvg.append(s('line', { x1: WB.x0, x2: WB.x0 + WB.w, y1: WB.y0 + WB.h, y2: WB.y0 + WB.h, stroke: 'var(--line)' }));
    for (const f of [0.25, 0.5]) {
      const y = WB.y0 + WB.h * (1 - f / 0.5);
      wsvg.append(s('line', { x1: WB.x0, x2: WB.x0 + WB.w, y1: y, y2: y, stroke: 'var(--grid)' }));
      wsvg.append(s('text', { x: WB.x0 - 4, y: y + 4, 'text-anchor': 'end' }, `${f * 100}%`));
    }
    BASIS.forEach(([, label], i) => {
      const share = weights[i] / tot;
      const bh = Math.max(0, Math.min(1, share / 0.5)) * WB.h;
      const x = WB.x0 + i * bw + 3;
      const bar = s('rect', { class: 'wbar' + (i === hover ? ' hl' : ''), x, y: WB.y0 + WB.h - bh, width: bw - 6, height: Math.max(bh, 1.5), rx: 3 });
      wsvg.append(bar);
      wsvg.append(s('text', { x: x + (bw - 6) / 2, y: WB.y0 + WB.h + 16, 'text-anchor': 'middle' }, label));
      if (share > 0.005) wsvg.append(s('text', { x: x + (bw - 6) / 2, y: WB.y0 + WB.h - bh - 4, 'text-anchor': 'middle', style: 'fill:var(--ink)' }, `${Math.round(share * 100)}`));
    });
    wsvg.append(s('text', { x: WB.x0 + WB.w / 2, y: WB.y0 + WB.h + 34, 'text-anchor': 'middle' }, 'stability s of each basis curve · weight in %'));
  }
  // Drag to set a bar: its share becomes the pointer height, the others keep their proportions.
  let dragging = false;
  function setFromPointer(ev) {
    const pt = wsvg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(wsvg.getScreenCTM().inverse());
    const i = Math.floor((p.x - WB.x0) / (WB.w / BASIS.length));
    if (i < 0 || i >= BASIS.length) return;
    const share = Math.max(0, Math.min(0.5, 0.5 * (WB.y0 + WB.h - p.y) / WB.h));
    const others = total() - weights[i];
    if (others <= 0) { weights[i] = 1; } else {
      // weights[i] / (others + weights[i]) = share
      weights[i] = share >= 0.999 ? others * 1000 : (share * others) / (1 - share);
    }
    hover = i;
    drawCurve();
  }
  wsvg.addEventListener('pointerdown', (ev) => { dragging = true; wsvg.setPointerCapture(ev.pointerId); setFromPointer(ev); });
  wsvg.addEventListener('pointermove', (ev) => {
    if (dragging) { setFromPointer(ev); return; }
    const pt = wsvg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(wsvg.getScreenCTM().inverse());
    const i = Math.floor((p.x - WB.x0) / (WB.w / BASIS.length));
    const nh = i >= 0 && i < BASIS.length ? i : -1;
    if (nh !== hover) { hover = nh; drawCurve(); }
  });
  wsvg.addEventListener('pointerup', () => { dragging = false; });
  wsvg.addEventListener('pointerleave', () => { if (!dragging && hover !== -1) { hover = -1; drawCurve(); } });

  // Curve plot.
  const P = { x0: 46, y0: 12, w: 496, h: 236 };
  const T0 = Math.log10(30), T1 = Math.log10(5 * 365 * DAY);
  const tx = (t) => P.x0 + P.w * (Math.log10(t) - T0) / (T1 - T0);
  const ty = (r) => P.y0 + P.h * (1 - r);
  const psvg = $('curve-plot');
  let probeT = null;
  const humanT = (t) => t < HOUR ? `${Math.round(t / MIN)} min` : t < DAY ? `${(t / HOUR).toFixed(1)} h` : t < 365 * DAY ? `${(t / DAY).toFixed(t < 10 * DAY ? 1 : 0)} days` : `${(t / 365 / DAY).toFixed(1)} years`;
  function path(f) {
    let d = '';
    for (let i = 0; i <= 200; i++) {
      const t = Math.pow(10, T0 + (T1 - T0) * i / 200);
      d += (i ? 'L' : 'M') + tx(t).toFixed(1) + ' ' + ty(f(t)).toFixed(1);
    }
    return d;
  }
  function drawPlot() {
    psvg.replaceChildren();
    const grid = s('g', { class: 'grid' });
    for (const r of [0, 0.25, 0.5, 0.75, 0.9, 1]) {
      grid.append(s('line', { x1: P.x0, x2: P.x0 + P.w, y1: ty(r), y2: ty(r) }));
      psvg.append(s('text', { x: P.x0 - 6, y: ty(r) + 4, 'text-anchor': 'end' }, `${Math.round(r * 100)}%`));
    }
    for (const [t, l] of [[MIN, '1 min'], [HOUR, '1 h'], [DAY, '1 day'], [7 * DAY, '1 week'], [30 * DAY, '1 month'], [365 * DAY, '1 year']]) {
      grid.append(s('line', { x1: tx(t), x2: tx(t), y1: P.y0, y2: P.y0 + P.h }));
      psvg.append(s('text', { x: tx(t), y: P.y0 + P.h + 16, 'text-anchor': 'middle' }, l));
    }
    psvg.prepend(grid);
    psvg.append(s('text', { x: P.x0 + P.w / 2, y: P.y0 + P.h + 34, 'text-anchor': 'middle' }, 'time since review (log scale) · probability of recall'));
    const tot = total();
    if (mode === 'basis') {
      BASIS.forEach((_, i) => {
        if (weights[i] / tot < 0.005 && i !== hover) return;
        psvg.append(s('path', { class: 'basis' + (i === hover ? ' hl' : ''), d: path((t) => R(t, i)) }));
      });
    } else {
      GRU.forEach((c, i) => psvg.append(s('path', { class: 'basis', d: path((t) => gruR(t, c)),
        style: `stroke:${GCOL[i]};stroke-opacity:.9;stroke-width:1.5;stroke-dasharray:4 3` })));
    }
    psvg.append(s('path', { class: 'mix', d: path(mix) }));
    if (probeT) {
      const r = mix(probeT);
      psvg.append(s('line', { class: 'probe', x1: tx(probeT), x2: tx(probeT), y1: P.y0, y2: P.y0 + P.h }));
      psvg.append(s('circle', { cx: tx(probeT), cy: ty(r), r: 5, fill: 'var(--s1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
      $('curve-read').textContent = `After ${humanT(probeT)}: ${(100 * r).toFixed(1)}% chance of recall.`;
    } else {
      // Report the interval at which recall falls to 90%, the usual Anki target.
      let lo = 1, hi = 1e10;
      for (let k = 0; k < 80; k++) { const mid = Math.sqrt(lo * hi); if (mix(mid) > 0.9) lo = mid; else hi = mid; }
      $('curve-read').textContent = `Recall falls to 90% after about ${humanT(lo)}. Hover the chart to read other delays.`;
    }
    if (mode === 'basis' && hover >= 0) $('curve-read').textContent += ` Highlighted: the ${BASIS[hover][1]} curve, weight ${Math.round(100 * weights[hover] / tot)}%.`;
  }
  psvg.addEventListener('pointermove', (ev) => {
    const pt = psvg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(psvg.getScreenCTM().inverse());
    if (p.x < P.x0 || p.x > P.x0 + P.w) { probeT = null; } else probeT = Math.pow(10, T0 + (T1 - T0) * (p.x - P.x0) / P.w);
    drawPlot();
  });
  psvg.addEventListener('pointerleave', () => { probeT = null; drawPlot(); });
  const logS0 = Math.log10(MIN), logS1 = Math.log10(5 * 365 * DAY);
  const weightOuts = [];
  GRU.forEach((c, i) => {
    const row = h('div', { class: 'gru-row', style: { '--fc': GCOL[i] } }, h('b', {}, h('i'), `curve ${i + 1}`));
    const slider = (label, min, max, step, value, set, show) => {
      const out = h('output');
      const inp = h('input', { type: 'range', min, max, step, value, 'aria-label': `curve ${i + 1} ${label}` });
      inp.addEventListener('input', () => { set(+inp.value); out.textContent = show(); weightOuts.forEach((f) => f()); drawPlot(); });
      out.textContent = show();
      return [h('label', {}, label, inp, out), out];
    };
    const [wl, wo] = slider('weight', 0, 1, 0.01, c.w, (v) => { c.w = v; }, () => `${Math.round(100 * c.w / gruTotal())}%`);
    weightOuts.push(() => { wo.textContent = `${Math.round(100 * c.w / gruTotal())}%`; });
    row.append(wl,
      slider('stability S', logS0, logS1, 0.01, Math.log10(c.S), (v) => { c.S = Math.pow(10, v); }, () => humanT(c.S))[0],
      slider('decay d', 0.1, 3, 0.05, c.d, (v) => { c.d = v; }, () => c.d.toFixed(2))[0]);
    $('curve-gru').append(row);
  });
  [...$('curve-mode').children].forEach((b) => b.addEventListener('click', () => {
    mode = b.dataset.mode;
    [...$('curve-mode').children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    $('curve-basis').hidden = mode !== 'basis';
    $('curve-gru').hidden = mode === 'basis';
    $('curve-note-basis').hidden = mode !== 'basis';
    $('curve-note-gru').hidden = mode === 'basis';
    hover = -1;
    drawCurve();
  }));
  function drawCurve() { drawWeights(); drawPlot(); }
  drawCurve();
})();

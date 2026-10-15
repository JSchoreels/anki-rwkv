// Architecture page. Every count is derived from the two configurations below, which mirror
// rwkv/architecture.py (Alex, default) and expertium/architecture.py (Expertium iter36), plus Expertium's current head (RWKV_GRU_HEAD=3, no correction head), which is the model shown as "Expertium".
(() => {
  'use strict';
  const { h, bindTip, fmt, esc } = window.UI;
  const $ = (id) => document.getElementById(id);

  // LoRA ranks are identical in both models: decay 16, in-context rate 16, value-residual 8, gate 16.
  const LORA = { d: 16, a: 16, v: 8, g: 16 };
  const STREAM_COLORS = { card: 'var(--s2)', deck: 'var(--s3)', note: 'var(--s4)', preset: 'var(--s5)', user: 'var(--s6)' };
  const STREAM_TEXT = {
    card: 'one card’s reviews, in order',
    deck: 'all reviews in the same deck',
    note: 'sibling cards of the same note',
    preset: 'all cards sharing a deck preset',
    user: 'every review the user made',
  };
  const MODELS = {
    alex: {
      name: 'Alex', full: 'Alex (default)', C: 128, H: 4, F: 4, headMult: 4, curves: 128, points: 128,
      streams: [['card', 3, 1.5], ['deck', 4, 2], ['note', 2, 1.5], ['preset', 3, 2], ['user', 4, 2]],
    },
    exp: {
      // Expertium's current runs: RWKV_GRU_HEAD=3 (3 learned power-law curves), correction head off.
      name: 'Expertium', full: 'Expertium', C: 32, H: 1, F: 4, headMult: 4, curves: 3, points: 0, gru: 3,
      streams: [['card', 1, 1], ['deck', 4, 1], ['note', 3, 1], ['preset', 3, 1], ['user', 3, 1]],
    },
    // Expertium-style config as trained in this repository (iter36 snapshot): 64 fixed curves + 64-point correction head.
    iter36: {
      name: 'Expertium iter36', full: 'Expertium-style, iter36 snapshot', C: 32, H: 1, F: 4, headMult: 4, curves: 64, points: 64,
      streams: [['card', 1, 1], ['deck', 4, 1], ['note', 3, 1], ['preset', 3, 1], ['user', 3, 1]],
    },
  };

  // Parameters of one RWKV-7 layer (time mixer + channel mixer), grouped for display.
  function layerParams(C, H, cmf) {
    const k = Math.floor(cmf * C);
    return {
      rkvo: 4 * C * C,
      ffn: 2 * C * k,
      decay: 2 * C * LORA.d + C,
      arate: 2 * C * LORA.a + C,
      gate: 2 * C * LORA.g,
      vres: 2 * C * LORA.v + C,
      small: 4 * C /* 2 norms */ + 9 * C /* 8 shift mixes + bonus */ + 2 * (C * H + H) /* k/v scales */ + 3 * C /* FFN norm + mix */,
    };
  }
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);

  function components(m) {
    const { C, H, F, headMult: hm, curves, points } = m;
    const Fd = F * C, hd = hm * C;
    const out = [{ id: 'enc', label: 'Input encoder', color: 'var(--s1)', n: 92 * Fd + Fd + 2 * Fd + Fd * C + C,
      desc: `92 → ${Fd} → ${C}, SiLU + LayerNorm` }];
    for (const [s, n, f] of m.streams) {
      const per = sum(layerParams(C, H, f));
      out.push({ id: s, label: `${s[0].toUpperCase()}${s.slice(1)} stream`, color: STREAM_COLORS[s], n: per * n,
        desc: `${n} layer${n > 1 ? 's' : ''} × ${fmt(per)} parameters; ${STREAM_TEXT[s]}` });
    }
    const instant = C * hd + hd + hd * 4 + 4;
    if (m.gru) {
      // prehead norm + 6 dummy 1x1 parameters kept by the code + weight head + (w, S, d) linears for N curves + Instant head
      out.push({ id: 'heads', label: 'Output heads', color: 'var(--s7)',
        n: 2 * C + 6 + (C * C + C + 2 * C + C * hd + hd) + 3 * (m.gru * hd + m.gru) + instant,
        desc: `${m.gru} learned power-law curves (weight, stability, decay each), no correction head, 4 grades` });
    } else {
      out.push({ id: 'heads', label: 'Output heads', color: 'var(--s7)',
        n: 2 * C + (C * hd + hd + hd * points + points) + (C * C + C + 2 * C + C * hd + hd + hd * curves + curves) + instant,
        desc: `${curves} basis curves, ${points} correction points, 4 grades` });
    }
    return out;
  }
  for (const m of Object.values(MODELS)) {
    m.comp = components(m);
    m.total = m.comp.reduce((a, c) => a + c.n, 0);
    m.layers = m.streams.reduce((a, s) => a + s[1], 0);
    m.layerBytes = (m.H * 32 * 32 + 2 * m.C) * 4; // WKV matrices + two token-shift vectors, fp32
  }
  const A = MODELS.alex, E = MODELS.exp, I = MODELS.iter36;
  const kib = (b) => (b / 1024).toLocaleString('en-US', { maximumFractionDigits: 2 }) + ' KiB';

  // ---------- Headline ----------
  $('h-params').textContent = `${(A.total / 1e6).toFixed(2)}M vs ${Math.round(E.total / 1e3)}k`;
  $('h-params-sub').textContent = `Alex ${fmt(A.total)} · Expertium ${fmt(E.total)} (${(A.total / E.total).toFixed(1)}× smaller)`;
  $('h-width').textContent = `${A.C}×${A.H} vs ${E.C}×${E.H}`;
  $('h-layers').textContent = `${A.layers} vs ${E.layers}`;
  $('h-layers-sub').textContent = `per stream: Alex ${A.streams.map((s) => s[1]).join('/')}, Expertium ${E.streams.map((s) => s[1]).join('/')} (card/deck/note/preset/user)`;
  $('h-mem').textContent = `${kib(A.layerBytes * A.streams[0][1]).replace(' KiB', '')} vs ${kib(E.layerBytes * E.streams[0][1])}`;

  // ---------- 92 input features ----------
  const G = {
    interval: { label: 'Interval since this card’s last review', color: 'var(--s1)',
      desc: 'Log days and log seconds since the previous review of this card, plus where that gap lands on a 24-hour clock (sin, cos).' },
    age: { label: 'Card age', color: 'var(--s2)',
      desc: 'Total time since the card was first reviewed (sum of all its intervals), in days, seconds and 24-hour phase.' },
    answer: { label: 'The answer itself', color: 'var(--s3)', hidden: true,
      desc: 'Grade (Again, Hard, Good, Easy as one-hot), answer duration and the dataset\u2019s review state. Blanked on the query row, which must predict the grade.' },
    load: { label: 'Workload and timing', color: 'var(--s4)',
      desc: 'Days since the user’s previous review, a 7-day cycle, new cards and other reviews since this card was last seen, and how many new cards and reviews were already done today.' },
    flags: { label: 'Flags', color: 'var(--s5)',
      desc: 'Whether the note, deck or preset is missing in the dataset, and whether this row is a query (grade hidden).' },
    ids: { label: 'Random ID tags', color: 'var(--s6)',
      desc: 'A random code per card (12), note (12), deck (8) and preset (8), redrawn every batch with values in {−1.5, −0.5, 0.5, 1.5}. The model can tell “the same card” apart from “another card” without memorising any real ID.' },
    phase: { label: 'Day-cycle phase, this review', color: 'var(--s7)',
      desc: 'Sin and cos of this review’s day in cycles of 3, 7, 30, 100, 365, 3,650 and 36,500 days. Days count from the start of the history window and each cycle gets a random offset per batch, so these encode relative timing, not calendar dates.' },
    phase1: { label: 'Day-cycle phase, card’s first day', color: 'var(--s8)',
      desc: 'The same seven cycles, with the same offsets, evaluated on the day this card was first reviewed.' },
  };
  const F = [
    ['scaled_elapsed_days', 'interval', 'log(1 + days since this card’s previous review), standardised'],
    ['scaled_elapsed_days_cumulative', 'age', 'log(1 + days since the card’s first review), standardised'],
    ['scaled_elapsed_seconds', 'interval', 'log(1 + seconds since this card’s previous review), standardised'],
    ['elapsed_seconds_sin', 'interval', 'gap modulo 24 h, sine'],
    ['elapsed_seconds_cos', 'interval', 'gap modulo 24 h, cosine'],
    ['scaled_elapsed_seconds_cumulative', 'age', 'log(1 + seconds since first review), standardised'],
    ['elapsed_seconds_cumulative_sin', 'age', 'card age modulo 24 h, sine'],
    ['elapsed_seconds_cumulative_cos', 'age', 'card age modulo 24 h, cosine'],
    ['scaled_duration', 'answer', 'log(10 + answer time in ms), standardised'],
    ['rating_1', 'answer', 'grade was Again'],
    ['rating_2', 'answer', 'grade was Hard'],
    ['rating_3', 'answer', 'grade was Good'],
    ['rating_4', 'answer', 'grade was Easy'],
    ['note_id_is_nan', 'flags', 'note unknown in the dataset'],
    ['deck_id_is_nan', 'flags', 'deck unknown in the dataset'],
    ['preset_id_is_nan', 'flags', 'preset unknown in the dataset'],
    ['day_offset_diff', 'load', 'log log of days since the user’s previous review of any card'],
    ['day_of_week', 'load', 'position in a 7-day cycle counted from the first day of this history window, scaled to [−1, 1] (not the calendar weekday)'],
    ['diff_new_cards', 'load', 'log of new cards introduced since this card was last seen'],
    ['diff_reviews', 'load', 'log of other reviews done since this card was last seen'],
    ['cum_new_cards_today', 'load', 'log of new cards already started today'],
    ['cum_reviews_today', 'load', 'log of reviews already done today'],
    ['scaled_state', 'answer', 'review-state code from the dataset (0 = learning start), shifted by −2'],
    ['is_query', 'flags', '1 on the query copy of a review'],
  ];
  for (const [id, n] of [['card', 12], ['note', 12], ['deck', 8], ['preset', 8]]) {
    for (let i = 0; i < n; i++) F.push([`${id}_tag[${i}]`, 'ids', `random ${id} code, digit ${i + 1} of ${n}`]);
  }
  for (const p of [3, 7, 30, 100, 365, 3650, 36500]) {
    F.push([`day_sin(${p})`, 'phase', `today in a ${fmt(p)}-day cycle, sine`]);
    F.push([`day_cos(${p})`, 'phase', `today in a ${fmt(p)}-day cycle, cosine`]);
    F.push([`first_day_sin(${p})`, 'phase1', `card’s first day in a ${fmt(p)}-day cycle, sine`]);
    F.push([`first_day_cos(${p})`, 'phase1', `card’s first day in a ${fmt(p)}-day cycle, cosine`]);
  }
  console.assert(F.length === 92, 'expected 92 input features');

  const strip = $('fstrip');
  const cellsByGroup = {};
  F.forEach(([name, g, desc], i) => {
    const c = h('div', { class: 'fcell' + (G[g].hidden ? ' hidden-q' : ''), style: { '--fc': G[g].color } });
    bindTip(c, `<b>#${i + 1} ${esc(name)}</b>${esc(desc)}<div class="m">${esc(G[g].label)}${G[g].hidden ? ' · hidden on query rows' : ''}</div>`);
    (cellsByGroup[g] = cellsByGroup[g] || []).push(c);
    strip.append(c);
  });
  for (const [g, def] of Object.entries(G)) {
    const n = cellsByGroup[g].length;
    const card = h('div', { class: 'fgroup', style: { '--fc': def.color } },
      h('b', {}, def.label, h('span', {}, `${n} number${n > 1 ? 's' : ''}`)), h('p', {}, def.desc));
    card.addEventListener('pointerenter', () => cellsByGroup[g].forEach((c) => c.classList.add('hl')));
    card.addEventListener('pointerleave', () => cellsByGroup[g].forEach((c) => c.classList.remove('hl')));
    $('fgroups').append(card);
  }

  // ---------- Pipeline ----------
  const UNIT = 256; // widest drawn quantity (Alex's feed-forward width) fills the column
  function pipeline(m) {
    const flow = h('div', { class: 'flow' });
    const enc = m.comp[0];
    flow.append(h('div', { class: 'sbar-label', style: { 'margin-top': '0' } }, m.full, h('span', {}, fmt(m.total) + ' parameters')));
    flow.append(h('div', { class: 'stage' }, h('div', { class: 't' }, '92 input numbers per review', h('span', {}, 'same in both')),
      h('div', { class: 'sub' }, 'plus a query copy of each review with the answer blanked')));
    flow.append(h('div', { class: 'arrow' }));
    flow.append(h('div', { class: 'stage stream', style: { '--fc': enc.color } },
      h('div', { class: 't' }, 'Input encoder', h('span', {}, fmt(enc.n))), h('div', { class: 'sub' }, enc.desc)));
    for (const [s, n, f] of m.streams) {
      const comp = m.comp.find((c) => c.id === s);
      flow.append(h('div', { class: 'arrow' }));
      const layers = h('div', { class: 'layers' });
      for (let i = 0; i < n; i++) {
        const bar = h('div', { class: 'layer', style: { width: (100 * m.C / UNIT) + '%', '--fc': comp.color } });
        for (let j = 0; j < m.H; j++) bar.append(h('i'));
        bindTip(bar, `<b>${comp.label}, layer ${i + 1}</b>${m.C} channels in ${m.H} head${m.H > 1 ? 's' : ''} of 32<div class="m">${fmt(sum(layerParams(m.C, m.H, f)))} parameters</div>`);
        const ffn = h('div', { class: 'layer cm', style: { width: (100 * Math.floor(f * m.C) / UNIT) + '%', '--fc': comp.color } });
        bindTip(ffn, `<b>Channel mixer</b>feed-forward width ${Math.floor(f * m.C)} (${f}× hidden width)`);
        layers.append(bar, ffn);
      }
      flow.append(h('div', { class: 'stage stream', style: { '--fc': comp.color } },
        h('div', { class: 't' }, comp.label, h('span', {}, `${n} × layer · ${fmt(comp.n)}`)),
        h('div', { class: 'sub' }, STREAM_TEXT[s]), layers));
    }
    flow.append(h('div', { class: 'arrow' }));
    const hd = m.headMult * m.C;
    flow.append(h('div', { class: 'heads3' },
      h('div', { class: 'stage stream', style: { '--fc': 'var(--s7)' } }, h('div', { class: 't' }, 'Curve mix'),
        h('div', { class: 'sub' }, m.gru ? `${m.C}→${m.C}→${hd}→ weight, stability and decay for ${m.curves} power-law curves` : `${m.C}→${m.C}→${hd}→ weights for ${m.curves} basis curves`)),
      m.gru ? h('div', { class: 'stage stream', style: { '--fc': 'var(--s7)' } }, h('div', { class: 't' }, 'No correction head'),
        h('div', { class: 'sub' }, 'switched off (RWKV_NO_AHEAD_RESIDUAL)'))
        : h('div', { class: 'stage stream', style: { '--fc': 'var(--s7)' } }, h('div', { class: 't' }, 'Curve correction'),
        h('div', { class: 'sub' }, `${m.C}→${hd}→ ${m.points} time points`)),
      h('div', { class: 'stage stream', style: { '--fc': 'var(--s7)' } }, h('div', { class: 't' }, 'Instant'),
        h('div', { class: 'sub' }, `${m.C}→${hd}→ 4 grades`))));
    return flow;
  }
  $('pipe').append(pipeline(A), pipeline(E));

  // ---------- Parameter waffles ----------
  const PER = 4000;
  // Largest-remainder rounding so the squares add up to the rounded total.
  function apportion(items, total) {
    const exact = items.map((it) => it.n / PER);
    const base = exact.map(Math.floor);
    let left = total - base.reduce((a, b) => a + b, 0);
    exact.map((x, i) => [x - Math.floor(x), i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left > 0) { base[i]++; left--; } });
    return base;
  }
  const waffleCells = {};
  function waffle(el, m, key) {
    const counts = apportion(m.comp, Math.round(m.total / PER));
    m.comp.forEach((c, i) => {
      for (let j = 0; j < counts[i]; j++) {
        const cell = h('i', { style: { '--fc': c.color } });
        bindTip(cell, `<b>${esc(c.label)}</b>${fmt(c.n)} parameters (${(100 * c.n / m.total).toFixed(1)}% of ${m.name})<div class="m">${esc(c.desc)}</div>`);
        (waffleCells[c.id] = waffleCells[c.id] || []).push(cell);
        el.append(cell);
      }
    });
    $(key + '-n').textContent = `${fmt(m.total)} ≈ ${Math.round(m.total / PER)} squares`;
  }
  waffle($('w-alex'), A, 'w-alex');
  waffle($('w-exp'), E, 'w-exp');
  $('w-exp-cap').textContent = `The whole Expertium model fits in the space Alex spends on its input encoder and part of its card stream.`;

  const allCells = () => Object.values(waffleCells).flat();
  A.comp.forEach((c) => {
    const key = h('span', { tabindex: '0' }, h('i', { style: { background: c.color } }), c.label);
    const on = () => allCells().forEach((x) => x.classList.toggle('dim', !(waffleCells[c.id] || []).includes(x)));
    const off = () => allCells().forEach((x) => x.classList.remove('dim'));
    key.addEventListener('pointerenter', on); key.addEventListener('pointerleave', off);
    key.addEventListener('focus', on); key.addEventListener('blur', off);
    $('pkeys').append(key);
  });

  for (const m of [A, E]) {
    const bar = h('div', { class: 'sbar' });
    m.comp.forEach((c) => {
      const seg = h('i', { style: { flex: String(c.n), '--fc': c.color } });
      bindTip(seg, `<b>${esc(c.label)}</b>${(100 * c.n / m.total).toFixed(1)}% of ${m.name} · ${fmt(c.n)}`);
      bar.append(seg);
    });
    $('shares').append(h('div', { class: 'sbar-label' }, m.full, h('span', {}, '100% = ' + fmt(m.total))), bar);
  }
  const pt = $('ptable');
  pt.append(h('thead', {}, h('tr', {}, h('th', {}, 'Component'), h('th', { class: 'r' }, 'Alex'), h('th', { class: 'r' }, 'share'),
    h('th', { class: 'r' }, 'Expertium'), h('th', { class: 'r' }, 'share'), h('th', { class: 'r' }, 'ratio'))));
  const tb = h('tbody');
  A.comp.forEach((c, i) => {
    const e = E.comp[i];
    tb.append(h('tr', {}, h('td', {}, h('span', { class: 'keys', style: { margin: '0' } }, h('span', {}, h('i', { style: { background: c.color } }), c.label))),
      h('td', { class: 'r' }, fmt(c.n)), h('td', { class: 'r' }, (100 * c.n / A.total).toFixed(1) + '%'),
      h('td', { class: 'r' }, fmt(e.n)), h('td', { class: 'r' }, (100 * e.n / E.total).toFixed(1) + '%'),
      h('td', { class: 'r' }, (c.n / e.n).toFixed(1) + '×')));
  });
  tb.append(h('tr', {}, h('td', { html: '<strong>Total</strong>' }), h('td', { class: 'r', html: `<strong>${fmt(A.total)}</strong>` }), h('td', { class: 'r' }, '100%'),
    h('td', { class: 'r', html: `<strong>${fmt(E.total)}</strong>` }), h('td', { class: 'r' }, '100%'), h('td', { class: 'r' }, (A.total / E.total).toFixed(1) + '×')));
  pt.append(tb);

  // ---------- Inside one layer ----------
  const LP = [
    ['rkvo', 'R, K, V, O matrices', 'var(--s1)'],
    ['ffn', 'Channel mixer', 'var(--s2)'],
    ['decay', 'Decay LoRA', 'var(--s3)'],
    ['arate', 'In-context rate LoRA', 'var(--s4)'],
    ['gate', 'Gate LoRA', 'var(--s5)'],
    ['vres', 'Value-residual LoRA', 'var(--s6)'],
    ['small', 'Norms, shift mixes, scales', 'var(--s7)'],
  ];
  LP.forEach(([, label, color]) => $('lkeys').append(h('span', {}, h('i', { style: { background: color } }), label)));
  function layerWaffle(el, m, stream) {
    const f = m.streams.find((s) => s[0] === stream)[2];
    const p = layerParams(m.C, m.H, f);
    const total = sum(p);
    const items = LP.map(([id, label, color]) => ({ id, label, color, n: p[id] }));
    // 100 squares = 100% of this layer
    const exact = items.map((it) => 100 * it.n / total);
    const base = exact.map(Math.floor);
    let left = 100 - base.reduce((a, b) => a + b, 0);
    exact.map((x, i) => [x - Math.floor(x), i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left > 0) { base[i]++; left--; } });
    el.replaceChildren();
    items.forEach((it, i) => {
      for (let j = 0; j < base[i]; j++) {
        const cell = h('i', { style: { '--fc': it.color } });
        bindTip(cell, `<b>${esc(it.label)}</b>${fmt(it.n)} parameters, ${(100 * it.n / total).toFixed(1)}% of this layer`);
        el.append(cell);
      }
    });
    return `${fmt(total)} parameters per layer (width ${m.C}, feed-forward ${Math.floor(f * m.C)})`;
  }
  let curStream = 'deck';
  function drawLayer() {
    $('l-alex-n').textContent = layerWaffle($('l-alex'), A, curStream);
    $('l-exp-n').textContent = layerWaffle($('l-exp'), E, curStream);
    [...$('layer-stream').children].forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.s === curStream)));
  }
  for (const s of ['card', 'deck', 'note', 'preset', 'user']) {
    const b = h('button', { type: 'button', 'data-s': s }, s);
    b.addEventListener('click', () => { curStream = s; drawLayer(); });
    $('layer-stream').append(b);
  }
  drawLayer();

  // ---------- State memory ----------
  function memory(el, m) {
    const size = 18; // same size in both models: every tile is one 32x32 matrix
    for (const [s, n] of m.streams) {
      const mats = h('div', { class: 'mats' });
      for (let l = 0; l < n; l++) {
        if (l) mats.append(h('span', { class: 'sep' }));
        for (let j = 0; j < m.H; j++) {
          const mat = h('span', { class: 'mat', style: { '--fc': STREAM_COLORS[s], '--ms': size + 'px' } });
          bindTip(mat, `<b>${s} stream · layer ${l + 1} · head ${j + 1}</b>32×32 state matrix = 1,024 numbers (4 KiB)`);
          mats.append(mat);
        }
      }
      const bytes = m.layerBytes * n;
      el.append(h('div', { class: 'mem' },
        h('div', { class: 'nm' }, `per ${s}`, h('small', {}, `${n} layer${n > 1 ? 's' : ''} × ${m.H} head${m.H > 1 ? 's' : ''}`)),
        mats, h('div', { class: 'kib' }, kib(bytes), h('small', {}, fmt(bytes / 4) + ' numbers'))));
    }
  }
  memory($('mem-alex'), A);
  memory($('mem-exp'), E);
  const headColors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
  for (const [id, m] of [['vec-alex', A], ['vec-exp', E]]) {
    for (let i = 0; i < m.C; i++) $(id).append(h('i', { style: { '--fc': headColors[Math.floor(i / 32)] } }));
  }

  // ---------- Output-head rugs (log time axis) ----------
  const LO = Math.log10(0.1), HI = Math.log10(Math.exp(22.2));
  const pos = (sec) => 100 * (Math.log10(sec) - LO) / (HI - LO);
  const space = (n, base, maxE) => Array.from({ length: n }, (_, i) =>
    base + (Math.exp(18.5 * i / (n - 1)) - 1) * Math.exp(maxE - 18.5));
  const TICKS = [[1, '1 s'], [60, '1 min'], [3600, '1 h'], [86400, '1 day'], [86400 * 30, '1 month'], [86400 * 365, '1 year'], [86400 * 3650, '10 years']];
  const human = (s) => s < 60 ? s.toFixed(1) + ' s' : s < 3600 ? (s / 60).toFixed(1) + ' min' : s < 86400 ? (s / 3600).toFixed(1) + ' h'
    : s < 86400 * 365 ? (s / 86400).toFixed(1) + ' days' : (s / 86400 / 365).toFixed(1) + ' years';
  function rug(el, values, color, what, withTicks) {
    values.forEach((v, i) => {
      if (v <= 0.1) v = 0.1;
      const t = h('i', { style: { left: pos(v) + '%', height: '100%', '--fc': color } });
      bindTip(t, `<b>${what} ${i + 1} of ${values.length}</b>${human(v)}`);
      el.append(t);
    });
    if (withTicks) TICKS.forEach(([s, l], i) => el.append(h('span', { class: 'tick' + (i % 2 ? '' : ' minor'), style: { left: pos(s) + '%' } }, l)));
  }
  // Expertium: three curves whose stability is predicted per review. The ticks mark only the starting
  // values (bias), log-spaced from 1 hour to 1 year (expertium/model/variant_model.py).
  {
    const n = E.gru;
    const init = Array.from({ length: n }, (_, i) => Math.exp(Math.log(3600) + (Math.log(31536000) - Math.log(3600)) * i / (n - 1)));
    $('rug-exp-n').textContent = `${n} learned power-law curves, head width ${E.headMult * E.C}; no correction head`;
    rug($('rug-exp-s'), init, 'var(--s1)', 'Initial stability of curve', true);
  }
  for (const [k, m] of [['alex', A], ['iter36', I]]) {
    $(`rug-${k}-n`).textContent = `${m.curves} basis curves (top) and ${m.points} correction points (bottom), head width ${m.headMult * m.C}`;
    rug($(`rug-${k}-s`), space(m.curves, 0.1, 22), 'var(--s1)', 'Basis curve', false);
    rug($(`rug-${k}-p`), space(m.points, 0.5, 21), 'var(--s2)', 'Correction point', true);
  }

  $('ahead-n').textContent = fmt(I.C * I.headMult * I.C + I.headMult * I.C + I.headMult * I.C * I.points + I.points);
  $('iter-total').textContent = fmt(I.total);

  // ---------- Comparison table ----------
  const rows = [
    ['Parameters', fmt(A.total), fmt(E.total)],
    ['Hidden width (d_model)', A.C, E.C],
    ['Heads (32 channels each)', A.H, E.H],
    ['Layers: card / deck / note / preset / user', A.streams.map((s) => s[1]).join(' / '), E.streams.map((s) => s[1]).join(' / ')],
    ['Channel-mixer factor', A.streams.map((s) => s[2]).join(' / '), E.streams.map((s) => s[2]).join(' / ')],
    ['LoRA ranks: decay / in-context rate / value / gate', '16 / 16 / 8 / 16', '16 / 16 / 8 / 16'],
    ['Input encoder', `92 → ${A.F * A.C} → ${A.C}`, `92 → ${E.F * E.C} → ${E.C}`],
    ['Curve head', `${A.curves} fixed exp. curves + ${A.points}-pt correction`, `${E.gru} learned power-law curves (weight, S, d per review), no correction head`],
    ['Head inner width', A.headMult * A.C, E.headMult * E.C],
    ['State per card', kib(A.layerBytes * A.streams[0][1]), kib(E.layerBytes * E.streams[0][1])],
    ['State per note', kib(A.layerBytes * A.streams[2][1]), kib(E.layerBytes * E.streams[2][1])],
    ['Dropout: card, note / deck, preset, user / per layer', '0.02 / 0.05 / 0.01', '0.02 / 0.05 / 0.01'],
  ];
  const ct = $('ctable');
  ct.append(h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', { class: 'r' }, 'Alex (default)'), h('th', { class: 'r' }, 'Expertium'))));
  const cb = h('tbody');
  rows.forEach(([k, a, e]) => cb.append(h('tr', {}, h('td', {}, k), h('td', { class: 'r' }, String(a)), h('td', { class: 'r' }, String(e)))));
  ct.append(cb);
})();

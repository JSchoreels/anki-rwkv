(() => {
  'use strict';
  // Overlapped vs sequential timeline for campaign.html. Durations are scaled from iteration 0090:
  // 238 ms per update (39.7 s / 167 updates), ~45.8k reviews per update, GPU ~61% busy, workers ~55% busy.
  const svg = document.getElementById('pl-tl');
  if (!svg) return;
  const NS = 'http://www.w3.org/2000/svg';
  const X0 = 96, X1 = 664, T = 1428, LH = 24, GAP = 10, Y0 = 34;
  const lanes = ['Worker 1', 'Worker 2', 'Worker 3', 'Worker 4', 'Worker 5', 'Main', 'GPU'];
  const C = { prep: '#1D9E75', main: '#7F77DD', gpu: '#D85A30', read: '#888780' };
  const sx = t => X0 + (X1 - X0) * t / T;
  const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const PREP = 650, DISPATCH = 188, GPU = 145, LAG = 45, READ = 12, UPDATE = 238, REVIEWS = 45800;

  function blocks(mode) {
    const b = [];
    if (mode === 'pipe') {
      for (let w = 0; w < 5; w++) for (let k = -1; k < 3; k++) {
        const s = (k * 5 + w) * UPDATE - 500;
        b.push({ lane: w, s: Math.max(0, s), e: Math.min(T, s + PREP), c: 'prep', lab: 'batch' });
      }
      for (let u = 0; u < 6; u++) {
        b.push({ lane: 5, s: u * UPDATE, e: u * UPDATE + DISPATCH, c: 'main', lab: 'u' + (u + 1) });
        b.push({ lane: 5, s: u * UPDATE + DISPATCH, e: u * UPDATE + DISPATCH + READ, c: 'read' });
        b.push({ lane: 6, s: u * UPDATE + LAG, e: u * UPDATE + LAG + GPU, c: 'gpu', lab: 'u' + (u + 1) });
      }
      return { b, ups: 1000 / UPDATE, gpu: GPU / UPDATE };
    }
    const step = PREP + LAG + GPU + READ;
    for (let t = 0, u = 0; t < T; u++) {
      b.push({ lane: 0, s: t, e: Math.min(T, t + PREP), c: 'prep', lab: 'batch' });
      t += PREP; if (t >= T) break;
      b.push({ lane: 5, s: t, e: Math.min(T, t + DISPATCH), c: 'main', lab: 'u' + (u + 1) });
      b.push({ lane: 6, s: t + LAG, e: Math.min(T, t + LAG + GPU), c: 'gpu', lab: 'u' + (u + 1) });
      t += LAG + GPU;
      b.push({ lane: 5, s: t, e: Math.min(T, t + READ), c: 'read' });
      t += READ;
    }
    return { b, ups: 1000 / step, gpu: GPU / step };
  }

  function render(mode) {
    svg.replaceChildren();
    for (let ms = 0; ms <= T; ms += 200) {
      const x = sx(ms);
      svg.appendChild(el('line', { x1: x, x2: x, y1: Y0 - 6, y2: Y0 + 7 * (LH + GAP) - GAP, stroke: 'var(--line)', 'stroke-width': 1 }));
      const t = el('text', { x, y: Y0 - 12, 'text-anchor': 'middle', class: 'pl-s' }); t.textContent = ms + ' ms'; svg.appendChild(t);
    }
    lanes.forEach((name, i) => {
      const t = el('text', { x: 8, y: Y0 + i * (LH + GAP) + LH / 2, 'dominant-baseline': 'central', class: i > 4 ? 'pl-t' : 'pl-s' });
      t.textContent = name; svg.appendChild(t);
      svg.appendChild(el('rect', { x: X0, y: Y0 + i * (LH + GAP), width: X1 - X0, height: LH, rx: 4, fill: 'var(--bg)', stroke: 'var(--line)' }));
    });
    const r = blocks(mode);
    r.b.forEach(k => {
      if (k.e <= k.s) return;
      const y = Y0 + k.lane * (LH + GAP);
      svg.appendChild(el('rect', { x: sx(k.s), y: y + 2, width: Math.max(1.5, sx(k.e) - sx(k.s)), height: LH - 4, rx: 3, fill: C[k.c] }));
      if (k.lab && sx(k.e) - sx(k.s) > 30) {
        const t = el('text', { x: (sx(k.s) + sx(k.e)) / 2, y: y + LH / 2, 'text-anchor': 'middle', 'dominant-baseline': 'central', fill: '#fff', 'font-size': 11 });
        t.textContent = k.lab; svg.appendChild(t);
      }
    });
    document.getElementById('pl-ups').textContent = r.ups.toFixed(1);
    document.getElementById('pl-gpu').textContent = Math.round(r.gpu * 100) + '%';
    document.getElementById('pl-rps').textContent = Math.round(r.ups * REVIEWS / 1000) + 'k';
    document.getElementById('pl-note').textContent = mode === 'pipe'
      ? 'Each worker needs about 650 ms per batch (u = update), but five of them work ahead in parallel, so a batch is always waiting. The main thread launches update n while the GPU still runs update n−1, and only reads the loss of update n−2. Block positions are illustrative; durations are scaled from iteration 0090.'
      : 'The same work done one step at a time: prepare, copy, compute, wait for the loss, repeat. The GPU idles while the CPU prepares the next batch. Same per-step durations as the overlapped view.';
    for (const [id, m] of [['pl-pipe', 'pipe'], ['pl-seq', 'seq']]) document.getElementById(id).setAttribute('aria-pressed', String(m === mode));
  }

  document.getElementById('pl-pipe').addEventListener('click', () => render('pipe'));
  document.getElementById('pl-seq').addEventListener('click', () => render('seq'));
  render('pipe');
})();

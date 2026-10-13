(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => (n == null ? '–' : Math.round(n).toLocaleString('en-US'));
  const pct = (n, d = 2) => (n == null ? '–' : (n > 0 ? '+' : '') + n.toFixed(d) + '%');
  const iterLabel = (n) => String(n).padStart(4, '0');
  const ORDER = ['accepted', 'stack_candidate', 'extend', 'investigate', 'fastmode', 'screen', 'rejected', 'no_comparison'];
  const POINTS = { accepted: 'rectRot', stack_candidate: 'triangle', extend: 'rect', rejected: 'circle', investigate: 'crossRot', fastmode: 'star', screen: 'circle', no_comparison: 'cross' };

  let DATA, charts = {}, hidden = new Set();
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const theme = () => ({
    ink: cssVar('--ink'), muted: cssVar('--muted'), grid: cssVar('--grid'), surface: cssVar('--surface'),
    accent: cssVar('--accent'), line: cssVar('--line'),
    c: (d) => cssVar('--c-' + d) || cssVar('--muted'),
  });

  fetch('data/r01.json', { cache: 'no-cache' })
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((d) => { DATA = d; init(); })
    .catch((e) => {
      document.querySelector('main').insertAdjacentHTML('afterbegin',
        '<div class="panel">Could not load <code>data/r01.json</code>: ' + String(e.message).replace(/</g, '&lt;') + '</div>');
    });

  function init() {
    DATA.exp = DATA.iterations.filter((e) => e.model === 'expertium');
    DATA.milestones.forEach((m, i) => { m.n = i + 1; });
    fillHeadline();
    buildLegend();
    buildTimelineFilters();
    buildTimeline();
    buildTable();
    buildCharts();
    $('theme').addEventListener('click', toggleTheme);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!document.documentElement.dataset.theme) rebuild(); });
    $('gen').textContent = 'Data generated ' + DATA.generated_utc + '. Cohort: ' + DATA.cohort + '.';
  }

  function toggleTheme() {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('r01-theme', next); } catch (e) { /* ignore */ }
    rebuild();
  }
  try { const t = localStorage.getItem('r01-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

  function rebuild() {
    Object.values(charts).forEach((c) => c.destroy());
    charts = {};
    buildLegend();
    buildCharts();
    buildTable();
  }

  function fillHeadline() {
    const h = DATA.headline;
    $('c-rps').textContent = fmt(h.accepted_reviews_per_second);
    $('c-rps-sub').textContent = 'reviews/s · accepted in iteration ' + iterLabel(h.accepted_iteration) + ' (' + h.accepted_track + ')';
    $('c-x').textContent = h.speedup_x ? '×' + h.speedup_x.toFixed(2) : '–';
    $('c-x-sub').textContent = 'vs ' + fmt(h.baseline_reviews_per_second) + ' reviews/s · ' + pct((h.speedup_x - 1) * 100, 0);
    $('h-speedup-title').textContent = h.speedup_x ? h.speedup_x.toFixed(1) + '×' : 'several times';
    $('c-it').textContent = h.latest_iteration;
    $('c-it-sub').textContent = h.iterations_counted + ' counted with a full comparison';
    const acc = DATA.exp.filter((e) => e.accepted).length;
    const stk = DATA.exp.filter((e) => e.decision === 'stack_candidate').length;
    const rej = DATA.exp.filter((e) => e.decision === 'rejected').length;
    $('c-acc').textContent = acc;
    $('c-acc-sub').textContent = stk + ' stack candidates · ' + rej + ' rejected';
  }

  function buildLegend() {
    const t = theme();
    const lg = $('legend');
    lg.textContent = '';
    const present = ORDER.filter((d) => DATA.exp.some((e) => e.decision === d));
    present.forEach((d) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip' + (hidden.has(d) ? ' off' : '');
      const i = document.createElement('i');
      i.style.background = t.c(d);
      b.append(i, DATA.decision_labels[d] || d);
      b.addEventListener('click', () => {
        hidden.has(d) ? hidden.delete(d) : hidden.add(d);
        b.classList.toggle('off');
        const ch = charts.rps;
        if (ch) {
          ch.data.datasets.forEach((ds) => { if (ds._decision) ds.hidden = hidden.has(ds._decision); });
          ch.update();
        }
      });
      lg.append(b);
    });
  }

  function milestoneAnnotations(t) {
    const ann = {};
    DATA.milestones.forEach((m) => {
      if (m.end_iteration) {
        ann['m' + m.n] = {
          type: 'box', xMin: m.iteration, xMax: m.end_iteration, backgroundColor: t.grid, borderWidth: 0,
          label: { display: true, content: 'plateau', position: { x: 'center', y: 'start' }, color: t.muted, font: { size: 11 } },
        };
      } else {
        ann['m' + m.n] = {
          type: 'line', xMin: m.iteration, xMax: m.iteration, borderColor: t.accent, borderDash: [4, 4], borderWidth: 1,
          label: {
            display: true, content: String(m.n), position: 'start', backgroundColor: t.accent, color: '#fff',
            font: { size: 10, weight: 'bold' }, padding: 2, borderRadius: 8,
          },
          enter: () => {}, leave: () => {},
        };
      }
    });
    return ann;
  }

  function tooltipLines(e) {
    const lines = [];
    lines.push('Iteration ' + iterLabel(e.iteration) + ' · ' + (e.label || e.track) + (e.track_name ? ' – ' + e.track_name : ''));
    let desc = e.description || '';
    if (desc.length > 230) desc = desc.slice(0, 227) + '…';
    wrap(desc, 56).forEach((l) => lines.push(l));
    lines.push('Decision: ' + (DATA.decision_labels[e.decision] || e.decision));
    lines.push('Full run: ' + fmt(e.reviews_per_second) + ' reviews/s' +
      (e.vs_baseline_x ? ' (×' + e.vs_baseline_x.toFixed(2) + ' baseline)' : ''));
    if (e.vs_previous_accepted_percent != null) lines.push('Single run vs previous accepted: ' + pct(e.vs_previous_accepted_percent));
    if (e.block) {
      lines.push('Blocks (' + e.block.n + '): mean ' + pct(e.block.mean_percent) + ', lower bound ' + pct(e.block.lower_bound_percent));
    }
    if (e.date) lines.push(e.date.slice(0, 16).replace('T', ' ') + ' UTC');
    return lines;
  }

  function wrap(s, w) {
    const out = []; let cur = '';
    s.split(/\s+/).forEach((word) => {
      if ((cur + ' ' + word).trim().length > w) { out.push(cur); cur = word; } else cur = (cur + ' ' + word).trim();
    });
    if (cur) out.push(cur);
    return out;
  }

  function baseOpts(t) {
    return {
      responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
      interaction: { mode: 'nearest', intersect: true },
      plugins: { legend: { display: false } },
      font: { family: 'system-ui, sans-serif' },
    };
  }

  function buildCharts() {
    const t = theme();
    Chart.defaults.color = t.muted;
    Chart.defaults.borderColor = t.grid;
    const exp = DATA.exp;
    const base = DATA.headline.baseline_reviews_per_second;
    const maxIt = Math.max(...exp.map((e) => e.iteration)) + 1;

    // 1. Throughput
    const steps = [{ x: 1, y: base }].concat(exp.map((e) => ({ x: e.iteration, y: e.accepted_reference_after })));
    const ds = [{
      type: 'line', label: 'Accepted reference', data: steps, stepped: 'after', borderColor: t.ink, borderWidth: 2,
      pointRadius: 0, pointHitRadius: 0, order: 5, _skipTip: true,
    }];
    ORDER.forEach((d) => {
      const pts = exp.filter((e) => e.decision === d && e.reviews_per_second).map((e) => ({ x: e.iteration, y: e.reviews_per_second, e }));
      if (!pts.length) return;
      ds.push({
        type: 'scatter', label: DATA.decision_labels[d], data: pts, _decision: d, hidden: hidden.has(d),
        backgroundColor: t.c(d), borderColor: d === 'accepted' ? t.surface : t.c(d), borderWidth: d === 'accepted' ? 1.5 : 1,
        pointStyle: POINTS[d], pointRadius: d === 'accepted' ? 6 : 4.5, pointHoverRadius: 8, order: d === 'accepted' ? 1 : 2,
      });
    });
    const o1 = baseOpts(t);
    o1.scales = {
      x: { type: 'linear', min: 0, max: maxIt, title: { display: true, text: 'Iteration' }, grid: { color: t.grid }, ticks: { stepSize: 10, precision: 0 } },
      y: { beginAtZero: true, title: { display: true, text: 'reviews/s' }, grid: { color: t.grid }, ticks: { callback: (v) => (v / 1000) + 'k' } },
    };
    o1.plugins.annotation = {
      annotations: Object.assign({
        base: {
          type: 'line', yMin: base, yMax: base, borderColor: t.muted, borderDash: [2, 4], borderWidth: 1,
          label: { display: true, content: 'original baseline ' + fmt(base), position: 'end', backgroundColor: 'transparent', color: t.muted, font: { size: 11 }, yAdjust: -10 },
        },
      }, milestoneAnnotations(t)),
    };
    o1.plugins.tooltip = {
      filter: (item) => !!item.raw.e,
      callbacks: { title: () => '', label: (item) => tooltipLines(item.raw.e) },
      bodyFont: { size: 12 }, padding: 10, displayColors: false,
    };
    charts.rps = new Chart($('ch-rps'), { data: { datasets: ds }, options: o1 });

    // 2. Cumulative speedup
    const sx = [{ x: 1, y: 1 }].concat(exp.map((e) => ({ x: e.iteration, y: e.accepted_reference_after / base, e })));
    const o2 = baseOpts(t);
    o2.scales = {
      x: { type: 'linear', min: 0, max: maxIt, title: { display: true, text: 'Iteration' }, grid: { color: t.grid }, ticks: { stepSize: 20, precision: 0 } },
      y: { beginAtZero: true, title: { display: true, text: 'speedup vs baseline' }, grid: { color: t.grid }, ticks: { callback: (v) => '×' + v } },
    };
    o2.interaction = { mode: 'nearest', axis: 'x', intersect: false };
    o2.plugins.tooltip = {
      callbacks: {
        title: (items) => 'Iteration ' + iterLabel(items[0].raw.x),
        label: (item) => '×' + item.raw.y.toFixed(2) + ' (' + fmt(item.raw.y * base) + ' reviews/s)',
        afterLabel: (item) => (item.raw.e && item.raw.e.accepted ? 'Accepted: ' + (item.raw.e.label || item.raw.e.track) : ''),
      },
    };
    charts.x = new Chart($('ch-x'), {
      type: 'line',
      data: { datasets: [{ data: sx, stepped: 'after', borderColor: t.accent, backgroundColor: t.accent + '22', fill: true, borderWidth: 2, pointRadius: 0, pointHoverRadius: 5 }] },
      options: o2,
    });

    // 3. Block gains
    const blk = exp.filter((e) => e.block && e.block.mean_percent != null);
    const labels = blk.map((e) => iterLabel(e.iteration));
    const o3 = baseOpts(t);
    o3.interaction = { mode: 'index', intersect: false };
    o3.scales = {
      x: { title: { display: true, text: 'Iteration (blocks measured)' }, grid: { display: false }, ticks: { maxRotation: 70, minRotation: 45, autoSkip: false, font: { size: 10 } } },
      y: { title: { display: true, text: 'per-update gain, %' }, grid: { color: t.grid } },
    };
    o3.plugins.tooltip = {
      callbacks: {
        title: (items) => { const e = blk[items[0].dataIndex]; return 'Iteration ' + iterLabel(e.iteration) + ' · ' + (e.label || e.track); },
        label: (item) => item.dataset.label + ': ' + pct(item.raw),
        afterBody: (items) => {
          const e = blk[items[0].dataIndex];
          return ['Blocks: ' + e.block.n + ' · verdict: ' + e.block.decision, DATA.decision_labels[e.decision] || ''];
        },
      },
    };
    charts.b = new Chart($('ch-b'), {
      data: {
        labels,
        datasets: [
          { type: 'bar', label: 'Mean gain', data: blk.map((e) => e.block.mean_percent), backgroundColor: blk.map((e) => t.c(e.decision)), borderRadius: 3, order: 2 },
          { type: 'line', label: 'Lower bound', data: blk.map((e) => e.block.lower_bound_percent), showLine: false, pointStyle: 'rectRot', pointRadius: 4, borderColor: t.ink, backgroundColor: t.surface, borderWidth: 1.5, order: 1 },
        ],
      },
      options: o3,
    });
  }

  let tlSel = new Set(['milestones']);

  function buildTimelineFilters() {
    const box = $('tl-filter');
    box.textContent = '';
    const mk = (key, label, color) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip' + (tlSel.has(key) ? '' : ' off');
      const i = document.createElement('i');
      i.style.background = color;
      b.append(i, label);
      b.addEventListener('click', () => {
        tlSel.has(key) ? tlSel.delete(key) : tlSel.add(key);
        if (!tlSel.size) tlSel.add('milestones');
        buildTimelineFilters();
        buildTimeline();
      });
      box.append(b);
    };
    mk('milestones', 'Milestones', 'var(--accent)');
    ORDER.filter((d) => DATA.exp.some((e) => e.decision === d)).forEach((d) => {
      const n = DATA.exp.filter((e) => e.decision === d).length;
      mk(d, (DATA.decision_labels[d] || d) + ' (' + n + ')', 'var(--c-' + d + ')');
    });
  }

  function buildTimeline() {
    const ol = $('timeline');
    ol.textContent = '';
    const byIt = {};
    DATA.exp.forEach((e) => { byIt[e.iteration] = e; });
    const items = new Map();
    if (tlSel.has('milestones')) {
      DATA.milestones.forEach((m) => {
        const e = byIt[m.iteration];
        items.set(m.iteration, { it: m.iteration, end: m.end_iteration, title: m.title, kind: m.kind, comment: m.comment, e: m.end_iteration ? null : e, m, n: m.n });
      });
    }
    DATA.exp.forEach((e) => {
      if (!tlSel.has(e.decision) || items.has(e.iteration)) return;
      let desc = e.description || '';
      if (desc.length > 320) desc = desc.slice(0, 317) + '…';
      items.set(e.iteration, { it: e.iteration, title: (e.label || e.track) + (e.track_name ? ' – ' + e.track_name : ''), comment: desc, e, n: null });
    });
    // a milestone card shows its decision colour only if that iteration's decision is selected or milestones are shown
    const list = [...items.values()].sort((a, b) => a.it - b.it);
    list.forEach((x) => {
      const e = x.e;
      const dec = e ? e.decision : 'none';
      const li = document.createElement('li');
      li.className = 'tl';
      li.style.setProperty('--dc', dec === 'none' ? 'var(--muted)' : 'var(--c-' + dec + ')');
      const dot = document.createElement('div'); dot.className = 'dot'; dot.textContent = x.n != null ? x.n : String(x.it).slice(-2);
      dot.title = 'Iteration ' + iterLabel(x.it);
      const body = document.createElement('div'); body.className = 'body';
      const h = document.createElement('h3'); h.textContent = x.title;
      const meta = document.createElement('div'); meta.className = 'meta';
      const span = (txt, cls) => { const s = document.createElement('span'); s.textContent = txt; if (cls) s.className = cls; return s; };
      const parts = [];
      if (e) parts.push(span(DATA.decision_labels[dec] || dec, 'tag dec-tag'));
      if (x.kind) parts.push(span(x.kind, 'tag neutral'));
      parts.push(span(x.end ? 'Iterations ' + iterLabel(x.it) + '–' + iterLabel(x.end) : 'Iteration ' + iterLabel(x.it)));
      if (e && e.reviews_per_second) parts.push(span(fmt(e.reviews_per_second) + ' reviews/s (×' + (e.vs_baseline_x || 0).toFixed(2) + ')'));
      if (e && e.block && e.block.mean_percent != null) parts.push(span('blocks: mean ' + pct(e.block.mean_percent) + ', lower ' + pct(e.block.lower_bound_percent)));
      else if (x.m && x.m.gain_vs_previous_accepted_percent != null) parts.push(span(pct(x.m.gain_vs_previous_accepted_percent, 1) + ' vs previous accepted (single run)'));
      meta.append(...parts);
      const p = document.createElement('p'); p.textContent = x.comment;
      body.append(h, meta, p);
      li.append(dot, body);
      ol.append(li);
    });
  }

  function buildTable() {
    const sel = $('filter');
    if (sel.options.length === 1) {
      ORDER.filter((d) => DATA.exp.some((e) => e.decision === d)).forEach((d) => {
        const o = document.createElement('option'); o.value = d; o.textContent = DATA.decision_labels[d]; sel.append(o);
      });
      sel.addEventListener('change', buildTable);
    }
    const t = theme();
    const tb = $('tbl').tBodies[0];
    tb.textContent = '';
    DATA.exp.slice().reverse().filter((e) => sel.value === 'all' || e.decision === sel.value).forEach((e) => {
      const tr = document.createElement('tr');
      const cell = (txt, cls) => { const td = document.createElement('td'); td.textContent = txt; if (cls) td.className = cls; tr.append(td); return td; };
      cell(iterLabel(e.iteration));
      const tk = cell(e.label || e.track); tk.title = e.description || '';
      cell(fmt(e.reviews_per_second), 'r');
      cell(e.vs_baseline_x ? '×' + e.vs_baseline_x.toFixed(2) : '–', 'r');
      const dc = document.createElement('td'); const sp = document.createElement('span'); sp.className = 'dec';
      const i = document.createElement('i'); i.style.background = t.c(e.decision);
      sp.append(i, DATA.decision_labels[e.decision] || e.decision); dc.append(sp); tr.append(dc);
      cell(e.block ? pct(e.block.mean_percent) : '–', 'r');
      cell(e.block ? pct(e.block.lower_bound_percent) : '–', 'r');
      tb.append(tr);
    });
  }
})();

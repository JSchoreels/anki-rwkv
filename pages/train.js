(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = n => n == null ? '–' : Math.round(n).toLocaleString('en-US');
  const pct = (a, b) => b ? (100 * a / b).toFixed(1) + '%' : '–';
  const time = iso => iso ? new Date(iso).toLocaleString(undefined, { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : '–';
  const dur = s => { if (s == null || s < 0) return '–'; const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; };

  function toggleTheme() {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('r01-theme', next); } catch (e) { /* ignore */ }
  }
  try { const t = localStorage.getItem('r01-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
  $('theme').addEventListener('click', toggleTheme);

  // Sample states for the preview; a real progress file has "mock": false and is rendered as is.
  function demoState(base, kind) {
    const d = JSON.parse(JSON.stringify(base));
    const n = d.training_users, E = d.epochs_planned;
    const start = Date.parse(d.started_utc), perEpoch = d.estimates.epoch_seconds, evalS = d.estimates.evaluation_seconds;
    const at = (e, frac) => new Date(start + (e - 1 + frac) * perEpoch * 1000).toISOString();
    d.epochs = [];
    if (kind === 'progress') {
      d.status = 'training';
      for (let e = 1; e <= E; e++) d.epochs.push({ epoch: e, users_done: e < 3 ? n : e === 3 ? 3150 : 0, finished_utc: e < 3 ? at(e, 1) : null, evaluation: null });
      d.updated_utc = at(3, 3150 / n);
    } else {
      d.status = kind === 'done' ? 'finished' : 'evaluating';
      for (let e = 1; e <= E; e++) d.epochs.push({ epoch: e, users_done: n, finished_utc: at(e, 1), evaluation: null });
      const last = d.epochs[E - 1];
      last.evaluation = kind === 'done'
        ? { users_done: d.evaluation_users, finished_utc: new Date(Date.parse(at(E, 1)) + evalS * 1000).toISOString(), log_loss: 0.3342, rmse_bins: 0.0418, auc: 0.7316 }
        : { users_done: 1240, finished_utc: null };
      d.updated_utc = kind === 'done' ? last.evaluation.finished_utc : new Date(Date.parse(at(E, 1)) + evalS * 620).toISOString();
    }
    return d;
  }

  function render(d) {
    const n = d.training_users, E = d.epochs_planned, slicePct = d.slice_percent || 5;
    const trainSlices = Math.round(80 / slicePct), evalSlices = Math.round(20 / slicePct);
    const usersPerSlice = n / trainSlices, evalPerSlice = d.evaluation_users / evalSlices;
    $('slice-pct').textContent = slicePct;

    const done = d.epochs.reduce((s, e) => s + Math.min(e.users_done || 0, n), 0);
    const total = n * E;
    const current = d.epochs.find(e => (e.users_done || 0) < n) || d.epochs[E - 1];
    const ev = d.epochs.map(e => e.evaluation).filter(Boolean);
    const lastEval = ev[ev.length - 1];

    const status = { training: 'Training', evaluating: 'Evaluating', finished: 'Finished', failed: 'Stopped', waiting: 'Not started' }[d.status] || d.status;
    $('c-status').textContent = status;
    $('c-status-sub').textContent = d.status === 'training' ? `epoch ${current.epoch} of ${E}`
      : d.status === 'evaluating' ? `held-out users ${fmt(lastEval && lastEval.users_done)} of ${fmt(d.evaluation_users)}`
      : d.status === 'finished' ? `${E} epochs, ${ev.length} evaluation${ev.length === 1 ? '' : 's'}` : '';
    $('c-progress').textContent = pct(done, total);
    $('c-progress-sub').textContent = `${fmt(done)} of ${fmt(total)} user-epochs`;
    $('c-rps').textContent = fmt(d.reviews_per_second);

    // ETA: remaining user-epochs at the measured epoch time, plus the planned evaluation passes still to run.
    let remaining = (total - done) / n * d.estimates.epoch_seconds;
    const evalLeft = (d.evaluate_after_epochs || []).filter(e => { const r = d.epochs[e - 1]; return !(r && r.evaluation && r.evaluation.finished_utc); });
    remaining += evalLeft.length * d.estimates.evaluation_seconds;
    if (d.status === 'evaluating' && lastEval) remaining -= (lastEval.users_done / d.evaluation_users) * d.estimates.evaluation_seconds;
    const finished = d.status === 'finished';
    $('c-eta').textContent = finished ? 'Done' : time(new Date(Date.parse(d.updated_utc) + remaining * 1000).toISOString());
    $('c-eta-sub').textContent = finished ? time(d.updated_utc) : `${dur(remaining)} left`;

    const rows = $('rows'); rows.innerHTML = '';
    d.epochs.forEach(e => {
      const row = document.createElement('div'); row.className = 'row'; row.setAttribute('role', 'listitem');
      const lbl = document.createElement('div'); lbl.className = 'lbl';
      lbl.innerHTML = `Epoch ${e.epoch}<small>${e.users_done >= n ? 'trained' : e.users_done ? pct(e.users_done, n) : '—'}</small>`;
      const train = document.createElement('div'); train.className = 'cells train';
      train.style.gridTemplateColumns = `repeat(${trainSlices}, 1fr)`;
      for (let i = 0; i < trainSlices; i++) {
        const c = document.createElement('div'); c.className = 'cell';
        const lo = i * usersPerSlice, hi = lo + usersPerSlice, u = e.users_done || 0;
        if (u >= hi) c.classList.add('trained');
        else if (u > lo) { c.classList.add('active'); c.style.setProperty('--fill', (100 * (u - lo) / usersPerSlice) + '%'); }
        c.title = `Epoch ${e.epoch}: users ${fmt(lo + 1)}–${fmt(hi)} ${u >= hi ? 'trained' : u > lo ? `training (${fmt(u - lo)} done)` : 'not yet'}`;
        train.appendChild(c);
      }
      const evc = document.createElement('div'); evc.className = 'cells eval';
      evc.style.gridTemplateColumns = `repeat(${evalSlices}, 1fr)`;
      const x = e.evaluation;
      for (let i = 0; i < evalSlices; i++) {
        const c = document.createElement('div'); c.className = 'cell';
        const lo = i * evalPerSlice, hi = lo + evalPerSlice, u = x ? (x.finished_utc ? d.evaluation_users : x.users_done || 0) : 0;
        if (u >= hi) c.classList.add('evaluated');
        else if (u > lo) { c.classList.add('active'); c.style.setProperty('--fill', (100 * (u - lo) / evalPerSlice) + '%'); }
        c.title = x ? `Held-out users ${fmt(lo + 1)}–${fmt(hi)} ${u >= hi ? 'evaluated' : u > lo ? 'evaluating' : 'waiting'}`
          : (d.evaluate_after_epochs || []).includes(e.epoch) ? 'Evaluation planned after this epoch' : 'Not evaluated after this epoch';
        evc.appendChild(c);
      }
      const m = document.createElement('div');
      if (x && x.finished_utc) {
        m.className = 'metrics';
        m.innerHTML = `<div class="m"><b>${x.log_loss.toFixed(4)}</b><span>LogLoss</span></div>` +
          `<div class="m"><b>${x.rmse_bins.toFixed(4)}</b><span>RMSE (bins)</span></div>` +
          `<div class="m"><b>${x.auc.toFixed(4)}</b><span>AUC</span></div>`;
      } else {
        m.className = 'metrics none';
        m.textContent = x ? `evaluating… ${pct(x.users_done || 0, d.evaluation_users)}`
          : (d.evaluate_after_epochs || []).includes(e.epoch) ? 'evaluation planned' : 'not evaluated';
      }
      row.append(lbl, train, evc, m);
      rows.appendChild(row);
    });
    $('rows-hint').textContent = `${trainSlices} training slices of ${fmt(usersPerSlice)} users and ${evalSlices} held-out slices of ${fmt(evalPerSlice)} users per row. Evaluation is planned after epoch ${(d.evaluate_after_epochs || []).join(', ') || '–'}.`;

    const det = [
      ['Model / configuration', d.model],
      ['Users', `${fmt(n)} training, ${fmt(d.evaluation_users)} held-out (disjoint)`],
      ['Training reviews per epoch', fmt(d.training_reviews_per_epoch)],
      ['Epochs planned', `${E} (evaluation after epoch ${(d.evaluate_after_epochs || []).join(', ')})`],
      ['Estimated epoch time', dur(d.estimates.epoch_seconds)],
      ['Estimated evaluation time', dur(d.estimates.evaluation_seconds)],
      ['Started', time(d.started_utc)],
      ['Checkpoints', d.checkpoints || '–'],
    ];
    $('details').querySelector('tbody').innerHTML = det.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
    $('updated').textContent = time(d.updated_utc);
  }

  fetch('data/train-8k.json', { cache: 'no-store' })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(base => {
      if (base.mock) {
        $('mock-bar').hidden = false;
        const sel = $('demo');
        const go = () => render(demoState(base, sel.value));
        sel.addEventListener('change', go); go();
      } else {
        render(base);
      }
    })
    .catch(err => { $('rows').textContent = `Could not load data/train-8k.json: ${err.message}`; });
})();

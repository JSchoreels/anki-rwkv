// Shared helpers for index.html and models.html: theme toggle and a hover tooltip.
(() => {
  'use strict';
  const root = document.documentElement;
  try { const t = localStorage.getItem('r01-theme'); if (t) root.dataset.theme = t; } catch (e) { /* ignore */ }

  const listeners = [];
  const btn = document.getElementById('theme');
  if (btn) {
    btn.addEventListener('click', () => {
      const cur = root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      try { localStorage.setItem('r01-theme', next); } catch (e) { /* ignore */ }
      listeners.forEach((f) => f());
    });
  }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => listeners.forEach((f) => f()));

  const tip = document.getElementById('tip');
  function show(html, ev) {
    if (!tip) return;
    tip.innerHTML = html;
    tip.classList.add('on');
    move(ev);
  }
  function move(ev) {
    if (!tip || !ev) return;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }
  function hide() { if (tip) tip.classList.remove('on'); }
  // Attach a tooltip to an element; html may be a string or a function returning one.
  function bindTip(el, html) {
    el.addEventListener('pointerenter', (ev) => show(typeof html === 'function' ? html() : html, ev));
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerleave', hide);
  }

  const fmt = (n) => Math.round(n).toLocaleString('en-US');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const h = (tag, attrs, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'style' && typeof v === 'object') Object.entries(v).forEach(([p, val]) => el.style.setProperty(p, val));
      else if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else el.setAttribute(k, v);
    }
    kids.flat().forEach((c) => c != null && el.append(c));
    return el;
  };

  window.UI = { onTheme: (f) => listeners.push(f), bindTip, hideTip: hide, fmt, esc, css, h };
})();

// Shared site menu. Reads nav.json (built by scripts/build_pages_nav.py from every page's <title> / nav metas, rerun on
// each Pages deploy) and fills the page's .eyebrow line with links to all pages, the current one unlinked. Pages without
// an .eyebrow get a small menu line at the top. If nav.json cannot be loaded the page keeps its static menu.
(() => {
  'use strict';
  const base = new URL('.', document.currentScript.src);
  const norm = (p) => p.replace(/index\.html$/, '').replace(/\/+$/, '/');
  const here = norm(location.pathname);

  function render(pages) {
    let line = document.querySelector('.eyebrow');
    if (!line) {
      line = document.createElement('p');
      line.className = 'eyebrow site-nav';
      line.style.cssText = 'margin:12px auto 0; padding:0 16px; max-width:1100px; font-size:13px; letter-spacing:.06em; text-transform:uppercase; opacity:.75';
      document.body.prepend(line);
    }
    line.replaceChildren();
    line.setAttribute('role', 'navigation');
    line.setAttribute('aria-label', 'Site pages');
    pages.forEach((p, i) => {
      if (i) line.append(' · ');
      const url = new URL(p.href, base);
      if (norm(url.pathname) === here) {
        const cur = document.createElement('span');
        cur.textContent = p.label;
        cur.setAttribute('aria-current', 'page');
        line.append(cur);
      } else {
        const a = document.createElement('a');
        a.href = url.href;
        a.textContent = p.label;
        line.append(a);
      }
    });
  }

  fetch(new URL('nav.json', base), { cache: 'no-cache' })
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((d) => { if (Array.isArray(d.pages) && d.pages.length) render(d.pages); })
    .catch(() => { /* keep the static menu */ });
})();

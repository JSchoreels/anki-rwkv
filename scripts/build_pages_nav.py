"""Build the shared site menu (nav.json) from the pages themselves, so every page links to every other page.

usage: build_pages_nav.py [SITE_DIR] [--inject]

Pages are the top-level *.html files of SITE_DIR plus the index.html of each subdirectory (for example
docs/data-quality/ on the deployed site). A page is left out when it is a redirect (<meta http-equiv="refresh">) or has
<meta name="nav" content="hide">. Its label is <meta name="nav-label"> or else its <title>; the order is
<meta name="nav-order"> (default 100), then label. --inject adds <script src=".../nav.js" defer> to listed pages that
lack it (used on the deploy copy, so a new page joins the menu without editing anything); a page whose
Content-Security-Policy blocks outside scripts gets the menu written in as static HTML instead. The Pages workflow runs
`build_pages_nav.py _site --inject`; run `build_pages_nav.py pages` locally to refresh the committed pages/nav.json.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path


def meta(text, name):
    m = re.search(rf'<meta\s+name="{name}"\s+content="([^"]*)"', text, re.I)
    return html.unescape(m.group(1)).strip() if m else None


def candidates(site):
    yield from sorted(site.glob("*.html"))
    yield from sorted(p for p in site.glob("**/index.html") if p.parent != site)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", nargs="?", default="pages")
    parser.add_argument("--inject", action="store_true", help="add the nav.js tag to pages that lack it")
    args = parser.parse_args()
    site = Path(args.site)
    pages = []
    for path in candidates(site):
        text = path.read_text(encoding="utf-8")
        if re.search(r'http-equiv="refresh"', text, re.I) or (meta(text, "nav") or "").lower() == "hide":
            continue
        title = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
        label = meta(text, "nav-label") or (html.unescape(title.group(1)).strip() if title else path.stem)
        rel = path.relative_to(site).as_posix()
        href = "./" if rel == "index.html" else rel[: -len("index.html")] if rel.endswith("/index.html") else rel
        order = meta(text, "nav-order")
        pages.append({"href": href, "label": label, "order": int(order) if order and order.lstrip("-").isdigit() else 100})
        pages[-1]["path"] = path
    pages.sort(key=lambda p: (p["order"], p["label"].lower()))
    for page in pages:
        path = page["path"]
        text = path.read_text(encoding="utf-8")
        if "nav.js" in text or "site-nav" in text:
            continue
        rel = path.relative_to(site).as_posix()
        if not args.inject:
            print(f"WARNING {rel} does not load nav.js (the deploy injects it)", file=sys.stderr)
            continue
        up = "../" * (len(path.relative_to(site).parts) - 1)
        if "Content-Security-Policy" in text:
            # outside scripts are blocked here: write the menu in as static HTML
            links = " &middot; ".join(
                f'<span aria-current="page">{html.escape(p["label"])}</span>' if p is page
                else f'<a href="{up}{"" if p["href"] == "./" else p["href"]}" style="color:inherit">{html.escape(p["label"])}</a>'
                for p in pages)
            block = ('<p class="eyebrow site-nav" role="navigation" aria-label="Site pages" style="margin:12px auto 0;'
                     ' padding:0 16px; max-width:1100px; font:13px system-ui,sans-serif; letter-spacing:.06em;'
                     f' text-transform:uppercase; opacity:.75">{links}</p>\n')
            text = re.sub(r"(<body[^>]*>\n?)", lambda m: m.group(1) + block, text, count=1)
        else:
            tag = f'<script src="{up}nav.js" defer></script>\n'
            text = text.replace("</body>", tag + "</body>", 1) if "</body>" in text else text + tag
        path.write_text(text, encoding="utf-8")
    out = site / "nav.json"
    out.write_text(json.dumps({"pages": [{"href": p["href"], "label": p["label"]} for p in pages]}, indent=1) + "\n",
                   encoding="utf-8")
    print(f"WROTE {out}: " + ", ".join(p["label"] for p in pages))


if __name__ == "__main__":
    main()

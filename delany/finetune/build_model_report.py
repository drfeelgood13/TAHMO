"""One page with the Ghana test-set diagnostics of several models and a model selector (top left).

Each model's report is built with build_report.py, then the pieces are merged: one copy of the styles and the
plot code, each model's content as an inert <template>, its data as JSON, and its table script as INIT[key]().
Choosing a model swaps its template into <main> and runs its script, so only one model is in the page at a time.

Usage (from the repo root): python finetune/build_model_report.py out.html
"""
import os
import re
import subprocess
import sys
import tempfile

FT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TITLE = "Ghana Model Comparison"

# key, button label, analysis folder, build_report.py arguments
MODELS = [
    ("combined", "Combined", "results/combined_ghana_conf/analysis",
     ["--model", "combined_1m", "--model-desc", "fine-tuned on Ghana + VIEDIA at 1 m", "--title", "Combined model (Ghana + VIEDIA)"]),
    ("ghana", "Ghana only", "results/ghana_1m/analysis",
     ["--model", "ghana_1m", "--model-desc", "fine-tuned on Ghana only at 1 m", "--title", "Ghana-only model"]),
    ("baseline", "Baseline", "results/zero_shot_1m/analysis",
     ["--model", "v2", "--tile-name", "v2 (no fine-tuning)", "--model-desc", "without fine-tuning, as released",
      "--title", "Baseline: v2 without fine-tuning"]),
]


def grab(pattern, text, what):
    m = re.search(pattern, text, re.S)
    if not m:
        raise ValueError(f"could not find {what}")
    return m


def main():
    out = sys.argv[1]
    parts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, label, analysis, extra in MODELS:
            path = os.path.join(tmp, key + ".html")
            subprocess.run([PY, os.path.join(FT, "build_report.py"), os.path.join(FT, analysis), path] + extra, check=True,
                           stdout=subprocess.DEVNULL)
            parts[key] = open(path).read()

    first = parts[MODELS[0][0]]
    head = first[:first.index("<main>")]
    head = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", head, count=1)
    xy_script = grab(r"(<script>\s*window\.makeXY.*?</script>)", first, "plot script").group(1)

    templates, data, inits = [], [], []
    for key, label, _, _ in MODELS:
        page = parts[key]
        inner = grab(r"<main>(.*)</main>", page, f"{key} main").group(1)
        img = grab(r'<script type="application/json" id="imgdata">(.*?)</script>', page, f"{key} image data").group(1)
        poly = grab(r'<script type="application/json" id="polydata">(.*?)</script>', page, f"{key} polygon data").group(1)
        table_js = grab(r"<script>(\s*\(\(\) => \{.*?const PAGE = 50;.*?\}\)\(\);\s*)</script>", page, f"{key} table script").group(1)
        table_js = table_js.replace("(() => {", f'window.INIT["{key}"] = () => {{', 1)
        table_js = table_js[:table_js.rindex("})();")] + "};" + table_js[table_js.rindex("})();") + 5:]
        table_js = (table_js.replace("document.getElementById('polydata')", f"document.getElementById('polydata-{key}')")
                            .replace('document.getElementById("imgdata")', f'document.getElementById("imgdata-{key}")'))
        assert f"polydata-{key}" in table_js and f"imgdata-{key}" in table_js, key
        templates.append(f'<template id="tpl-{key}">{inner}</template>')
        data.append(f'<script type="application/json" id="imgdata-{key}">{img}</script>\n'
                    f'<script type="application/json" id="polydata-{key}">{poly}</script>')
        inits.append(f"<script>{table_js}</script>")

    buttons = "".join(f'<button type="button" role="radio" id="pick-{k}" data-key="{k}" aria-checked="false">{lab}</button>'
                      for k, lab, _, _ in MODELS)
    bar_css = """
<style>
.modelbar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); border-bottom: 1px solid var(--rule);
  margin-inline: -20px; padding: 10px 20px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; }
.modelbar .lbl { font: 500 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.models { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; background: var(--surface); }
.models button { font: 600 13.5px var(--f-body); color: var(--muted); background: none; border: 0; padding: 7px 14px; cursor: pointer; }
.models button + button { border-left: 1px solid var(--rule); }
.models button[aria-checked="true"] { background: var(--accent); color: var(--surface); }
.models button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.modelbar .ctx { font-size: 13px; color: var(--muted); }
body { padding-block-start: 0; }
main#view { padding-top: 32px; }
</style>"""
    switch_js = """<script>
(() => {
  const KEYS = %s, view = document.getElementById("view");
  function show(key, push) {
    if (!KEYS.includes(key)) key = KEYS[0];
    view.innerHTML = document.getElementById("tpl-" + key).innerHTML;
    window.INIT[key]();
    KEYS.forEach(k => document.getElementById("pick-" + k).setAttribute("aria-checked", k === key));
    try { localStorage.setItem("ghana-model", key); } catch (e) {}
    if (push) { try { history.replaceState(null, "", "#" + key); } catch (e) {} }
  }
  document.querySelector(".models").addEventListener("click", e => {
    const b = e.target.closest("button"); if (b) { show(b.dataset.key, true); window.scrollTo(0, 0); } });
  document.querySelector(".models").addEventListener("keydown", e => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const cur = KEYS.findIndex(k => document.getElementById("pick-" + k).getAttribute("aria-checked") === "true");
    const next = KEYS[(cur + (e.key === "ArrowRight" ? 1 : KEYS.length - 1)) %% KEYS.length];
    show(next, true); document.getElementById("pick-" + next).focus(); });
  let start = location.hash.slice(1);
  if (!KEYS.includes(start)) { try { start = localStorage.getItem("ghana-model") || ""; } catch (e) { start = ""; } }
  show(start, false);
})();
</script>""" % ("[" + ",".join(f'"{k}"' for k, _, _, _ in MODELS) + "]")

    page = (head + bar_css + "\n"
            + f'<div class="modelbar"><span class="lbl">Model</span><div class="models" role="radiogroup" aria-label="Model">{buttons}</div>'
              f'<span class="ctx">Ghana test set · 32 images · 1 m</span></div>\n'
            + '<main id="view"></main>\n' + "\n".join(templates) + "\n" + xy_script + "\n" + "\n".join(data) + "\n"
            + "window.INIT = {};".join(["<script>", "</script>"]) + "\n" + "\n".join(inits) + "\n" + switch_js + "\n")
    open(out, "w").write(page)
    print("written", out, len(page))


if __name__ == "__main__":
    main()

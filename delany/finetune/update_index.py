"""Add DelAny V2 model (3), Ghana + Viedia with the backbone unfrozen, to the report in finetune/index.html,
and number the fine-tuned models: Ghana (1), Ghana + Viedia (2), Ghana + Viedia unfrozen (3).

Steps:
  1. rename the fine-tuned models everywhere (text, charts, scripts);
  2. rebuild the tile-level mAP section of every DelAny V2 page with four models (build_report.py);
  3. add model (3): its page (template), its data, its table script, and it joins Experiment 2;
  4. update the written text of the report with model (3)'s results (TEXT_EDITS, filled in from its scores).
Reads index.html, writes index.html (keep a backup). Run from the repo root:  python finetune/update_index.py
"""
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

FT = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(FT, "index.html")
PY = sys.executable

DISPLAY = {"baseline": "DelAny V2", "ghana": "DelAny V2: Ghana (1)", "combined": "DelAny V2: Ghana + Viedia (2)",
           "unfrozen": "DelAny V2: Ghana + Viedia unfrozen (3)"}
SHORT = {"baseline": "DelAny V2", "ghana": "Ghana (1)", "combined": "Ghana + Viedia (2)", "unfrozen": "Ghana + Viedia unfrozen (3)"}
OLD = {"ghana": ("DelAny V2: Ghana-Trained", "Ghana-Trained"), "combined": ("DelAny V2: Ghana Viedia Trained", "Ghana Viedia Trained")}
# build_report.py arguments per DelAny V2 page
PAGES = {
    "baseline": ("results/zero_shot_1m/analysis", ["--model", "v2", "--model-desc", "without fine-tuning, as released"]),
    "ghana": ("results/ghana_1m/analysis", ["--model", "ghana_1m", "--model-desc", "fine-tuned on Ghana only at 1 m"]),
    "combined": ("results/combined_ghana_conf/analysis", ["--model", "combined_1m", "--model-desc", "fine-tuned on Ghana + VIEDIA at 1 m"]),
    "unfrozen": ("results/unfrozen_ghana/analysis", ["--model", "combined_1m_unfrozen_mosaic", "--model-desc",
                 "fine-tuned on Ghana + VIEDIA at 1 m with the whole network trained (backbone unfrozen) and mosaic augmentation"]),
}
CODE = {k: v[1][1] for k, v in PAGES.items()}

# tile-level scores on the Ghana 1 m test tiles (Ultralytics validation; finetune/runs/delany-test_*.log, tile_experiments/scores.csv)
TILES = pd.DataFrame([
    ("baseline", 0.294, 0.191, 0.147, 0.063, 0.182, 0.088),
    ("ghana", 0.339, 0.303, 0.219, 0.092, 0.255, 0.118),
    ("combined", 0.384, 0.326, 0.249, 0.107, 0.290, 0.136),
    ("unfrozen", 0.4045, 0.3452, 0.2720, 0.1187, 0.3102, 0.1511),
], columns=["key", "mask_p", "mask_r", "mask_map50", "mask_map50_95", "box_map50", "box_map50_95"])


def build(key, tile_csv, tmp):
    analysis, extra = PAGES[key]
    out = os.path.join(tmp, key + ".html")
    subprocess.run([PY, os.path.join(FT, "build_report.py"), os.path.join(FT, analysis), out, *extra,
                    "--title", DISPLAY[key], "--tile-name", SHORT[key], "--tile-scores", tile_csv], check=True, stdout=subprocess.DEVNULL)
    page = open(out).read()
    # the report names models, not code names, in the mAP sentence
    page = page.replace(f"{CODE[key]} raises mask mAP50", f"{DISPLAY[key]} raises mask mAP50").replace("(v2 without fine-tuning)", "(DelAny V2)")
    return page


def section(html, title):
    m = re.search(rf"<section>\s*<h2>{re.escape(title)}</h2>.*?</section>", html, re.S)
    assert m, title
    return m


def table_script(page, key):
    js = re.search(r"<script>(\s*\(\(\) => \{.*?const PAGE = 50;.*?\}\)\(\);\s*)</script>", page, re.S).group(1)
    js = js.replace("(() => {", f'window.INIT["{key}"] = () => {{', 1)
    j = js.rindex("})();")
    js = js[:j] + "};" + js[j + 5:]
    return (js.replace("document.getElementById('polydata')", f"document.getElementById('polydata-{key}')")
              .replace('document.getElementById("imgdata")', f'document.getElementById("imgdata-{key}")'))


def rep(s, a, b, count=1):
    assert s.count(a) >= count, f"not found ({s.count(a)}x): {a[:90]}"
    return s.replace(a, b) if count == "all" else s.replace(a, b, count)


def main():
    s = open(IDX).read()
    assert 'id="tpl-unfrozen"' not in s, "model (3) is already in the report"

    # 1. names
    for key, (full, short) in OLD.items():
        s = s.replace(full, DISPLAY[key])
    s = rep(s, "The Viedia-trained model has slightly fewer false polygons", "Model (2) has slightly fewer false polygons")
    s = rep(s, "the Ghana-trained model covers a little more area", "model (1) covers a little more area")

    with tempfile.TemporaryDirectory() as tmp:
        tile_csv = os.path.join(tmp, "tiles.csv")
        TILES.assign(model=TILES.key.map(SHORT)).assign(mask_f1=lambda d: 2 * d.mask_p * d.mask_r / (d.mask_p + d.mask_r)) \
            .drop(columns="key").to_csv(tile_csv, index=False)
        pages = {k: build(k, tile_csv, tmp) for k in PAGES}

    # 2. mAP section of the existing DelAny V2 pages
    for key in ("baseline", "ghana", "combined"):
        t = re.search(rf'<template id="tpl-{key}">(.*?)</template>', s, re.S)
        old_sec = section(t.group(1), "Tile-level accuracy (mAP)")
        new_sec = section(re.search(r"<main>(.*)</main>", pages[key], re.S).group(1), "Tile-level accuracy (mAP)").group(0)
        tpl = t.group(1)[:old_sec.start()] + new_sec + t.group(1)[old_sec.end():]
        s = s[:t.start(1)] + tpl + s[t.end(1):]
    for key, (full, short) in OLD.items():
        assert short not in s, f"old name left: {short}"

    # 3. model (3): page, data, table script, navigation
    u = pages["unfrozen"]
    inner = re.search(r"<main>(.*)</main>", u, re.S).group(1)
    img = re.search(r'<script type="application/json" id="imgdata">(.*?)</script>', u, re.S).group(1)
    poly = re.search(r'<script type="application/json" id="polydata">(.*?)</script>', u, re.S).group(1)
    end_combined = s.index("</template>", s.index('<template id="tpl-combined">')) + len("</template>")
    s = s[:end_combined] + f'\n<template id="tpl-unfrozen">{inner}</template>' + s[end_combined:]
    s = rep(s, "<!-- ftp-ftw:data:begin -->",
            f'<script type="application/json" id="imgdata-unfrozen">{img}</script>\n'
            f'<script type="application/json" id="polydata-unfrozen">{poly}</script>\n<!-- ftp-ftw:data:begin -->')
    s = rep(s, "<!-- ftp-ftw:init:begin -->", f"<script>{table_script(u, 'unfrozen')}</script>\n<!-- ftp-ftw:init:begin -->")
    s = rep(s, '''                  {key: "combined", name: "DelAny V2: Ghana + Viedia (2)", col: "var(--s3)", cls: "c3"},''',
            '''                  {key: "combined", name: "DelAny V2: Ghana + Viedia (2)", col: "var(--s3)", cls: "c3"},
                  {key: "unfrozen", name: "DelAny V2: Ghana + Viedia unfrozen (3)", col: "var(--s6)", cls: "c6"},''')
    s = rep(s, 'combined: "DelAny V2: Ghana + Viedia (2)", ftp: "FTP"', 'combined: "DelAny V2: Ghana + Viedia (2)", unfrozen: "DelAny V2: Ghana + Viedia unfrozen (3)", ftp: "FTP"')
    s = rep(s, 'combined: "1 m imagery", ftp:', 'combined: "1 m imagery", unfrozen: "1 m imagery", ftp:')
    s = rep(s, 'keys: ["baseline", "ghana", "combined"], ctx:', 'keys: ["baseline", "ghana", "combined", "unfrozen"], ctx:')
    s = rep(s, 'window.INIT["ft"] = makeCompare({keys: ["baseline", "ghana", "combined"], target: 2, pair: [1, 2]});',
            'window.INIT["ft"] = makeCompare({keys: ["baseline", "ghana", "combined", "unfrozen"], target: 3, pair: [2, 3]});')

    # 4. written text
    for a, b in TEXT_EDITS:
        s = rep(s, a, b)

    open(IDX, "w").write(s)
    print("written", IDX, len(s))


TEXT_EDITS = [
    # Home: Experiment 2 card
    ("Three versions of the same model on 1 m imagery.",
     "Four versions of the same model on 1 m imagery: the model as released and three fine-tuned versions, numbered (1) to (3)."),
    ('<li><i class="sw3" style="background:var(--s3)"></i>DelAny V2: Ghana + Viedia (2)</li>',
     '<li><i class="sw3" style="background:var(--s3)"></i>DelAny V2: Ghana + Viedia (2)</li>\n        <li><i class="sw3" style="background:var(--s6)"></i>DelAny V2: Ghana + Viedia unfrozen (3)</li>'),
    # Home: abstract
    ("We compare five field-delineation models", "We compare six field-delineation models"),
    ("Three models are variants of Delineate Anything v2", "Four models are variants of Delineate Anything v2"),
    ("the model fine-tuned on the Ghana labels (<em>DelAny V2: Ghana (1)</em>) and the model fine-tuned on the Ghana labels plus the VIEDIA data from Vietnam and Cambodia (<em>DelAny V2: Ghana + Viedia (2)</em>).",
     "the model fine-tuned on the Ghana labels (<em>DelAny V2: Ghana (1)</em>), the model fine-tuned on the Ghana labels plus the VIEDIA data from Vietnam and Cambodia (<em>DelAny V2: Ghana + Viedia (2)</em>), and the same with the whole network trained and mosaic augmentation (<em>DelAny V2: Ghana + Viedia unfrozen (3)</em>)."),
    ("Adding VIEDIA changes little. Used as released",
     "Adding VIEDIA on its own changes little, but training the whole network on Ghana and VIEDIA (model 3) is a clear further step: mean F1 per image 0.22, 32% of fields found one-to-one, a boundary error of about 20 m, and tile-level mask mAP@0.5 0.272 against 0.249. Used as released"),
    ('<div class="tl-v">0.12 → 0.19</div><div class="tl-d">▲ after fine-tuning on Ghana</div><div class="tl-n">DelAny V2 → DelAny V2: Ghana + Viedia (2)</div>',
     '<div class="tl-v">0.12 → 0.22</div><div class="tl-d">▲ after fine-tuning on Ghana</div><div class="tl-n">DelAny V2 → DelAny V2: Ghana + Viedia unfrozen (3)</div>'),
    ('<div class="tl-v">107 → 23 m</div>', '<div class="tl-v">107 → 20 m</div>'),
    ('<div class="tl-v">12% → 27%</div>', '<div class="tl-v">12% → 32%</div>'),
    # Home: research questions, models table
    ('<span class="m">Both</span></li>\n  </ol>',
     '<span class="m">Both</span></li>\n    <li><b>Does training the whole network help?</b> Does fine-tuning every layer, with mosaic augmentation, improve on fine-tuning only the later layers? <span class="m">Experiment 2</span></li>\n  </ol>'),
    ('<td class="t">Ghana labels + VIEDIA (Vietnam, Cambodia)</td><td class="t">1 m RGB</td><td class="t">Experiment 2</td></tr>',
     '<td class="t">Ghana labels + VIEDIA (Vietnam, Cambodia)</td><td class="t">1 m RGB</td><td class="t">Experiment 2</td></tr>\n'
     '    <tr><td class="t"><span class="mk"><i class="sw3" style="background:var(--s6)"></i>DelAny V2: Ghana + Viedia unfrozen (3)</span></td><td class="t">The same as (2), but with the whole network trained (backbone unfrozen, half the learning rate) and mosaic augmentation; the best epoch is chosen by validation mask mAP@0.5</td><td class="t">Ghana labels + VIEDIA (Vietnam, Cambodia)</td><td class="t">1 m RGB</td><td class="t">Experiment 2</td></tr>'),
    ('The five models. Colours are the same in every chart.', 'The six models. Colours are the same in every chart.'),
    # Experiment 2: header
    ("Three DelAny V2 models compared</h1>", "Four DelAny V2 models compared</h1>"),
    ("how three versions of the same field-delineation model", "how four versions of the same field-delineation model"),
    (", and <b>DelAny V2: Ghana + Viedia (2)</b> is fine-tuned on the Ghana labels plus VIEDIA (data from Vietnam and Cambodia).",
     ", <b>DelAny V2: Ghana + Viedia (2)</b> is fine-tuned on the Ghana labels plus VIEDIA (data from Vietnam and Cambodia), and <b>DelAny V2: Ghana + Viedia unfrozen (3)</b> is trained on the same data with the whole network unfrozen and mosaic augmentation."),
    ('<span class="mk"><i class="sw3" style="background:var(--s3)"></i>DelAny V2: Ghana + Viedia (2)</span>.',
     '<span class="mk"><i class="sw3" style="background:var(--s3)"></i>DelAny V2: Ghana + Viedia (2)</span>\n  <span class="mk"><i class="sw3" style="background:var(--s6)"></i>DelAny V2: Ghana + Viedia unfrozen (3)</span>.'),
    # Experiment 2: summary
    ("and only 15–18% match a field.", "and only 15–20% match a field."),
    ("<li><b>Field size is still the main weakness.</b>",
     "<li><b>Training the whole network helps further.</b> DelAny V2: Ghana + Viedia unfrozen (3) is the best model: mean F1 per image 0.22 against 0.19 for (2), 32% of fields found one-to-one against 27%, fewer fields merged with a neighbour (25% against 31%) and a boundary error of about 20 m. It beats (2) on F1 on 21 of 32 images. On tiles, mask mAP@0.5 rises from 0.249 to 0.272.</li>\n    <li><b>Field size is still the main weakness.</b>"),
    # 1 Headline results
    ("Scorecard for the three models.", "Scorecard for the four models."),
    ("Both fine-tuned models trade precision for recall.", "All three fine-tuned models trade precision for recall."),
    # 2 Reference fields
    ("missed fields fall to 11–14% and merged fields to about 32%", "missed fields fall to 11–15% and merged fields to 25–32%"),
    ("they cover 78–82% of field area, against 64%", "they cover 77–82% of field area, against 64%"),
    ("(15% for DelAny V2: Ghana (1)) match a field", "(15% for DelAny V2: Ghana (1), 20% for DelAny V2: Ghana + Viedia unfrozen (3)) match a field"),
    # 3 Size and crop
    ("In all three models the smallest quarter", "In all four models the smallest quarter"),
    ("where the found share goes from 15–19% to 38–45%.", "where the found share goes from 15–19% to 38–45%, and to 47–53% for DelAny V2: Ghana + Viedia unfrozen (3)."),
    ("but are split about 40% of the time by both fine-tuned models.", "but are split about 40% of the time by all three fine-tuned models."),
    ("49% for DelAny V2: Ghana + Viedia (2)),", "49% for DelAny V2: Ghana + Viedia (2), 51% for DelAny V2: Ghana + Viedia unfrozen (3)),"),
    ("improve two- to four-fold after fine-tuning (14–30%)", "improve two- to four-fold after fine-tuning (14–36%)"),
    # 4 Image by image
    ("Between the two fine-tuned models there is no consistent winner (DelAny V2: Ghana + Viedia (2) is ahead on F1 on 19 of 32 images).",
     "Between models (1) and (2) there is no consistent winner (DelAny V2: Ghana + Viedia (2) is ahead on F1 on 19 of 32 images). Model (3) is more consistent: DelAny V2: Ghana + Viedia unfrozen (3) beats DelAny V2 on F1 for 27 of 32 images and model (2) for 21."),
    ("where the fine-tuned models typically reach about 22 m.", "where the fine-tuned models typically reach about 22 m (about 17 m for model (3))."),
    # 5 Drivers
    ("0.38 for DelAny V2: Ghana + Viedia (2))", "0.38 for DelAny V2: Ghana + Viedia (2), 0.45 for DelAny V2: Ghana + Viedia unfrozen (3))"),
    ("(ρ = −0.46 and −0.36)", "(ρ = −0.46, −0.36 and −0.45)"),
    # 6 Polygons
    ("In all three models the line is flat", "In all four models the line is flat"),
    # 7 Discussion
    ("<b>Why VIEDIA adds little.</b> The two fine-tuned models are indistinguishable within the noise of 32 images.",
     "<b>Why VIEDIA adds little.</b> Models (1) and (2) are indistinguishable within the noise of 32 images."),
    ("<p class=\"prose\"><b>Limitations.</b></p>",
     "<p class=\"prose\"><b>Why training the whole network helps.</b> Model (3) differs from model (2) only in how it is trained: all layers instead of only the later ones, at half the learning rate, with mosaic augmentation. It is the only change in this experiment that helps on tiles and on full images, and it does so on most images (21 of 32 better than model (2)). Adapting the early layers, which recognise basic edges and textures, to 1 m dry-season imagery seems to matter more than adding more training data.</p>\n  <p class=\"prose\"><b>Limitations.</b></p>"),
    # 8 Conclusions
    ("Adding VIEDIA does not measurably improve it, and small and large fields remain the weak spots.",
     "Adding VIEDIA alone does not measurably improve it; training the whole network on Ghana and VIEDIA (model 3) does. Small and large fields remain the weak spots."),
    ("Image difficulty follows field density and the share of non-crop land.</li>",
     "Image difficulty follows field density and the share of non-crop land.</li>\n    <li><b>RQ5, does training the whole network help? Yes.</b> DelAny V2: Ghana + Viedia unfrozen (3) is the best model: mean F1 per image 0.22 against 0.19, fields found one-to-one 32% against 27%, boundary error about 20 m against 23 m, and tile-level mask mAP@0.5 0.272 against 0.249. It is better than model (2) on 21 of 32 images.</li>"),
    ("perform the same within the noise of 32 images.</li>",
     "perform the same within the noise of 32 images.</li>\n      <li>Training the whole network (model 3) is the one change that clearly helps further: about +16% mean F1 per image over model (2).</li>"),
    ("Absolute scores are still low (F1 below 0.2 at IoU 0.5)", "Absolute scores are still low (mean F1 per image at most 0.22 at IoU 0.5)"),
    ("Use DelAny V2: Ghana + Viedia (2) as the default model, unless extra coverage matters more than a clean output.",
     "Use DelAny V2: Ghana + Viedia unfrozen (3) as the default model: it is the best on F1, fields found and boundary error."),
    ("before choosing between the two fine-tuned models.", "before treating small differences between the fine-tuned models as real."),
    # model (3) is the headline model of Experiment 2: summary tiles, Table 2's last column and the pair plots follow it
    ("the last column shows the change from DelAny V2 to DelAny V2: Ghana + Viedia (2)", "the last column shows the change from DelAny V2 to DelAny V2: Ghana + Viedia unfrozen (3)"),
    ("F1 per image: DelAny V2 (X) against DelAny V2: Ghana + Viedia (2) (Y). Above the dashed line DelAny V2: Ghana + Viedia (2) is better.",
     "F1 per image: DelAny V2 (X) against DelAny V2: Ghana + Viedia unfrozen (3) (Y). Above the dashed line DelAny V2: Ghana + Viedia unfrozen (3) is better."),
    ("(axes reversed so that up and right are better). Above the dashed line DelAny V2: Ghana + Viedia (2) is better.",
     "(axes reversed so that up and right are better). Above the dashed line DelAny V2: Ghana + Viedia unfrozen (3) is better."),
]

if __name__ == "__main__":
    main()

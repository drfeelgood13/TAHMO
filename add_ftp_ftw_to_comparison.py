"""Add the FTP and FTW model pages (and their data) to index.html.

Usage:  python add_ftp_ftw_to_comparison.py

Scores the polygons in ftp/output/test_set_b7 and ftw/output/test_set with
comparison_metrics.py, then writes, between marker comments in the HTML:
  * the <template> for each model page,
  * the embedded per-image and per-polygon data (same shape as the other models),
  * the script that drives those two pages.
Running it again replaces those blocks. The compare page, the model bar and the
page switcher are edited in the HTML itself.
"""
import html as htmllib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import comparison_metrics as cm
from evaluation import ROOT

PAGE = ROOT / "index.html"
MARK = "<!-- ftp-ftw:{name}:{edge} -->"

MODELS = {
    "ftp": dict(name="FTP", results=ROOT / "ftp/output/test_set_b7", id_column="field_id",
                result_path="ftp/output/test_set_b7/&lt;image&gt;/polygons.parquet"),
    "ftw": dict(name="FTW", results=ROOT / "ftw/output/test_set", id_column="id",
                result_path="ftw/output/test_set/&lt;image&gt;_0p3m/polygons.parquet"),
}
CROP_ORDER = ["Fallow", "Maize", "Other crops", "Rice", "Unidentified"]
CONF_ORDER = [("high", "high"), ("medium", "medium"), ("low", "low"), ("none", "not given")]
OUTCOMES = ["Found", "Poor outline", "Split", "Merged", "Missed"]
PRED_CATS = [("Matched field", "Matches one reference field (IoU ≥ 0.5)"),
             ("Piece of a field", "Mostly inside reference fields, but no match: a piece of a split field or a loose outline"),
             ("On non-crop background", "At least half inside the non-crop background polygon"),
             ("Covers several fields", "Covers two or more reference fields (merged)"),
             ("Mixed / boundary", "Straddles fields and background")]
ORDINAL = ["first", "second", "third", "fourth"]


# ---------------------------------------------------------------- formatting
def esc(s):
    return htmllib.escape(str(s), quote=False)


def pct(v):
    return f"{int(v * 100 + 0.5)}%"


def pct_or_dash(v):
    return "–" if v is None else pct(v)


def read_html():
    """The page uses CRLF line endings; work on LF and convert back when writing."""
    return PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n")


def write_html(html):
    PAGE.write_bytes(html.replace("\n", "\r\n").encode("utf-8"))


def embedded(html, element_id):
    return json.loads(re.search(rf'<script type="application/json" id="{element_id}">(.*?)</script>', html, re.S).group(1))


def bars(items):
    rows = []
    for i, (label, share) in enumerate(items):
        y, w = 8 + 26 * i, share * 260
        rows.append(f'<text x="140" y="{y + 11}" text-anchor="end" class="bl">{esc(label)}</text>'
                    f'<rect x="150" y="{y}" width="260" height="14" rx="2" class="track"/>'
                    f'<rect x="150" y="{y}" width="{w:.1f}" height="14" rx="2" class="bar"><title>{esc(label)}: {pct(share)}</title></rect>'
                    f'<text x="{150 + w + 6:.1f}" y="{y + 11}" class="bv">{pct(share)}</text>')
    return f'<svg class="bars" viewBox="0 0 480 {26 * len(items) + 6}" role="img" preserveAspectRatio="xMinYMin meet">{"".join(rows)}</svg>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + f'<td class="t">{r[0]}</td>' + "".join(f'<td class="n">{c}</td>' for c in r[1:]) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------- statistics
def field_frame(poly, flags):
    f = pd.DataFrame(poly["fields"], columns=cm.FIELD_COLS)
    f["split"] = [s for s, _ in flags]
    f["merged"] = [m for _, m in flags]
    f["found"] = f.outcome == "Found"
    f["size_q"] = pd.qcut(f.area.rank(method="first"), 4, labels=False)
    return f


def group_row(label, g):
    return [label, f"{len(g):,}", pct(g.found.mean()), pct(g["cov"].mean()), f"{g.iou.median():.2f}", pct(g.split.mean()), pct(g.merged.mean())]


def quartile_table(f, column, title, label_fmt, edge_fmt):
    q = pd.qcut(f[column].rank(method="first"), 4, labels=False)
    cuts = np.quantile(f[column], [0, .25, .5, .75, 1])
    rows = []
    for k in range(4):
        g = f[q == k]
        rows.append([f"Q{k + 1}: {edge_fmt(cuts[k])}–{edge_fmt(cuts[k + 1])}", f"{len(g):,}", pct(g.found.mean()), pct(g["cov"].mean()),
                     pct(g.split.mean()), pct(g.merged.mean())])
    return (f'<div class="fig"><h3>{title}</h3>' + table([label_fmt, "Fields", "Found", "Covered", "Split", "Merged"], rows) + "</div>"), \
        [r[2] for r in rows], [r[3] for r in rows]


def stats_for(key, img, poly, flags, summary, locs):
    f = field_frame(poly, flags)
    p = pd.DataFrame(poly["preds"], columns=["img", "id", "cat", "area", "in_f", "in_nc", "k", "col", "vari", "tex", "edge", "comp"])
    cols = [c[0] for c in img["cols"]]
    i = pd.DataFrame(img["rows"], columns=cols)
    return dict(f=f, p=p, i=i, summary=summary, locs=locs)


def pooled_row(poly):
    f = pd.DataFrame(poly["fields"])
    tp, n_pred, n_gt = int((f[6] == "Found").sum()), len(poly["preds"]), len(f)
    return tp, n_pred, tp / n_pred, tp / n_gt, 2 * tp / (n_gt + n_pred)


# ---------------------------------------------------------------- page text
def model_texts(key, S):
    """Model description, key findings and advice. Numbers come from the scored data."""
    f, p, i, sm = S["f"], S["p"], S["i"], S["summary"]
    n_pred = len(p)
    found = int(f.found.sum())
    no_poly = int((i.Predicted == 0).sum())
    one_poly = int((i.Predicted == 1).sum())
    f1_zero = int((i.F1 == 0).sum())
    cat = p.cat.value_counts(normalize=True)
    quarters = [f[f.size_q == k] for k in range(4)]
    rice = f[f.crop == "Rice"]
    several = p[p.cat == "Covers several fields"]
    t = {}
    if key == "ftp":
        t["eyebrow"] = "FTP-PRUE+ · EfficientNet-B7 · PlanetScope 3 m · Ghana test set"
        t["h1"] = "FTP model (PlanetScope, 3 m)"
        t["lede"] = ("What the reference fields and the predicted fields look like, and which of their characteristics go with good or poor delineation. "
                     "Model: FTP-PRUE+ with an EfficientNet-B7 backbone (<code>taylor-geospatial/ftp-b7</code>) as released, without any fine-tuning on Ghana, "
                     "run on two PlanetScope acquisitions per image at about 3 m. Its polygons are scored against the same 1,534 reference fields as the Delineate Anything models.")
        t["model_dl"] = (
            "<dt>Model</dt><dd>FTP-PRUE+ (EfficientNet-B7). A semantic-segmentation network that labels every pixel as background, field interior or field boundary. Used as released; no Ghana training.</dd>"
            "<dt>Input</dt><dd>PlanetScope surface reflectance (4 bands, about 3 m) from two time windows per image: April–May 2025 and August–September 2025. The reference labels come from the 2021 field survey.</dd>"
            "<dt>Polygons</dt><dd>The class map is turned into polygons with a watershed (<code>ftp/run_ftp_prue.py</code>). Polygons below 0.05 ha (500 m²) are dropped. The script notes that this vectorisation approximates, and is not identical to, the paper's own post-processing.</dd>"
            "<dt>Scoring</dt><dd>As for every model on this page: labels and predictions are clipped to the image box, reference fields under 20 m² are excluded, and a field counts as found when one polygon matches it one-to-one at IoU ≥ 0.5.</dd>")
        t["findings"] = [
            f"<b>It finds almost no field.</b> Only {found} of 1,534 reference fields ({pct(found / len(f))}) are matched one-to-one at IoU ≥ 0.5, and the mean F1 per image is {i.F1.mean():.3f}. "
            f"The fields it does match fit reasonably well (mean IoU {sm['mean_iou_of_matches']:.2f}); matches are just rare.",
            f"<b>It draws few, very large polygons.</b> {n_pred} polygons for 1,534 fields. {no_poly} images get no polygon at all and {one_poly} get exactly one; "
            f"the polygons that cover several fields are {pct(cat['Covers several fields'])} of the output and have a median size of {several.area.median():,.0f} m².",
            f"<b>Merging, not splitting, is the main error.</b> {pct(f.merged.mean())} of the reference fields lie inside a polygon that also covers another field, and only {pct(f.split.mean())} are cut into pieces. "
            f"On average {pct(f['cov'].mean())} of a field's area lies under some polygon, yet one-to-one matches stay at {pct(f.found.mean())}.",
            f"<b>Field size works against it.</b> None of the smallest quarter of fields (under about 740 m²) is matched; even the largest quarter reaches only {pct(quarters[3].found.mean())}. "
            f"With 3 m pixels and a 500 m² minimum polygon size, fields of this size are at or below what the run can output.",
            f"<b>Rice is the one crop it covers well.</b> {pct(rice['cov'].mean())} of a rice field's area is covered and {pct(rice.found.mean())} of rice fields are found, the highest of any crop, "
            f"but {pct(rice.merged.mean())} of rice fields are merged with a neighbour. Fallow and \"other crops\" are covered least ({pct(f[f.crop == 'Fallow']['cov'].mean())} and {pct(f[f.crop == 'Other crops']['cov'].mean())}).",
            f"<b>{pct(cat['On non-crop background'])} of the polygons fall on the non-crop background</b> ({int((p.cat == 'On non-crop background').sum())} of {n_pred}), "
            f"{pct(cat['Piece of a field'])} are pieces inside fields, and only {pct(cat['Matched field'])} match a field.",
            f"<b>Most images score zero.</b> {f1_zero} of 32 images have F1 = 0, {no_poly} of them because there is no polygon. Boundary quality can only be judged on the {32 - no_poly} images with polygons; there the mean boundary error (PoLiS) is "
            f"{i['PoLiS (m)'].mean():.0f} m, against about 107 m for DelAny V2 and 23 m for the fine-tuned models."]
        t["suggests"] = [
            "<b>This run is not usable as it stands.</b> About one field in fifty is matched, so the 30 matches are not a result to build on.",
            "<b>Check the inputs first.</b> The imagery is from 2025 and the labels from 2021, and the model was not fine-tuned on Ghana. Check a few images in QGIS to see whether the class map shows field boundaries at all; if the boundary class is weak, the watershed will merge fields.",
            "<b>Try the official post-processing.</b> The run script says its watershed is an approximation. The big merged polygons suggest the boundary class is not separating neighbouring fields, so the paper's own post-processing and different seed settings (<code>--h-seed</code>) are worth one run.",
            "<b>Expect a size limit.</b> At 3 m, fields under about 740 m² are only a few dozen pixels. Compare with a higher-resolution input before judging the model on small fields."]
    else:
        t["eyebrow"] = "FTW-PRUE · EfficientNet-B7 · Sentinel-2 10 m · Ghana test set"
        t["h1"] = "FTW model (Sentinel-2, 10 m)"
        t["lede"] = ("What the reference fields and the predicted fields look like, and which of their characteristics go with good or poor delineation. "
                     "Model: the released Fields of The World model <code>FTW_PRUE_EFNET_B7_CCBY</code>, without any fine-tuning on Ghana, "
                     "run on two Sentinel-2 acquisitions per image at 10 m. Its polygons are scored against the same 1,534 reference fields as the Delineate Anything models.")
        t["model_dl"] = (
            "<dt>Model</dt><dd>FTW-PRUE with an EfficientNet-B7 backbone (<code>FTW_PRUE_EFNET_B7_CCBY</code>), used as released. It takes a pair of Sentinel-2 scenes and predicts field interiors and boundaries; no Ghana training.</dd>"
            "<dt>Input</dt><dd>Two Sentinel-2 scenes per image from Microsoft Planetary Computer, picked from the 2024 crop-calendar start and end dates (first within ±14 days and below 20% cloud, then progressively wider and cloudier, up to ±180 days). The reference labels come from the 2021 field survey.</dd>"
            "<dt>Polygons</dt><dd>The predicted raster is cropped back to the image box and turned into polygons (<code>ftw/run_ftw_coordinates.py</code>).</dd>"
            "<dt>Scoring</dt><dd>As for every model on this page: labels and predictions are clipped to the image box, reference fields under 20 m² are excluded, and a field counts as found when one polygon matches it one-to-one at IoU ≥ 0.5.</dd>")
        t["findings"] = [
            f"<b>It finds almost no field.</b> Only {found} of 1,534 reference fields ({pct(found / len(f))}) are matched one-to-one at IoU ≥ 0.5, and the mean F1 per image is {i.F1.mean():.3f}. "
            f"Mean IoU of the fields it does match is {sm['mean_iou_of_matches']:.2f}.",
            f"<b>Most fields are simply missed.</b> {pct((f.outcome == 'Missed').mean())} of the reference fields have less than 20% of their area under any polygon; on average only {pct(f['cov'].mean())} of a field is covered. "
            f"{no_poly} images get no polygon at all. This is the opposite of FTP, which draws a few huge polygons that cover much of the land.",
            f"<b>What it draws is small.</b> {n_pred} polygons for 1,534 fields; {pct(cat['Piece of a field'])} are pieces inside fields (median {p[p.cat == 'Piece of a field'].area.median():,.0f} m²), "
            f"only {pct(cat['Matched field'])} match a field and {pct(cat['Covers several fields'])} cover several. Merging is rare ({pct(f.merged.mean())} of fields) and so is splitting ({pct(f.split.mean())}).",
            f"<b>Size hardly matters because almost nothing is found.</b> The share found goes {pct(quarters[0].found.mean())}, {pct(quarters[1].found.mean())}, {pct(quarters[2].found.mean())}, {pct(quarters[3].found.mean())} from the smallest to the largest quarter, "
            f"and the largest quarter is the only one where fields are split to any extent ({pct(quarters[3].split.mean())}). A 10 m pixel is 100 m², so a 740 m² field is only about seven pixels.",
            f"<b>Rice is again the easiest crop.</b> {pct(rice.found.mean())} of rice fields are found and {pct(rice['cov'].mean())} of their area is covered, against {pct(f[f.crop == 'Maize'].found.mean())} found and {pct(f[f.crop == 'Maize']['cov'].mean())} covered for maize.",
            f"<b>{pct(cat['On non-crop background'])} of the polygons fall on the non-crop background</b> ({int((p.cat == 'On non-crop background').sum())} of {n_pred}).",
            f"<b>Most images score zero.</b> {f1_zero} of 32 images have F1 = 0, {no_poly} of them because there is no polygon. Boundary quality can only be judged on the {32 - no_poly} images with polygons; there the mean boundary error (PoLiS) is "
            f"{i['PoLiS (m)'].mean():.0f} m, about the same as DelAny V2 (107 m) and well above the fine-tuned models (23 m)."]
        t["suggests"] = [
            "<b>This run is not usable as it stands.</b> About one field in eighty-five is matched, so the 18 matches are not a result to build on.",
            "<b>Check the dates.</b> The Sentinel-2 scenes are from 2024 and the labels from a 2021 survey, and the dates were chosen from a crop calendar rather than from each image. A scene from the wrong part of the season, or a cloudy one (the search was relaxed to 100% cloud cover when nothing else was found), could explain the empty output on some images; check the scenes for the images with no polygons.",
            "<b>Look at the raw prediction.</b> The raster the model writes (<code>inference_output.tif</code>) shows whether it sees field interiors and boundaries at all, or whether the polygon step throws them away.",
            "<b>Expect a size limit.</b> Fields under about 740 m² are only a few Sentinel-2 pixels wide. This model is better suited to larger, more regular fields than most of these smallholder plots."]
    return t


def size_text(S):
    f = S["f"]
    q = [f[f.size_q == k] for k in range(4)]
    best = int(np.argmax([g.found.mean() for g in q]))
    s_found = q[0].found.mean()
    small = ("are never found (0%)" if s_found == 0 else
             f"are found only {pct(s_found)} of the time" if s_found >= 0.005 else "are found less than 1% of the time")
    return (f"Fields under about 740 m² {small}; {pct(q[0].merged.mean())} of them end up inside a larger prediction. "
            f"Fields are found most often in the {ORDINAL[best]} size quarter ({pct(q[best].found.mean())}). In the largest quarter, {pct(q[3].split.mean())} of fields are cut into several pieces "
            f"and {pct(q[3].merged.mean())} are merged with neighbours. Area covered by any polygon is {pct(q[0]['cov'].mean())}, {pct(q[1]['cov'].mean())}, {pct(q[2]['cov'].mean())} and {pct(q[3]['cov'].mean())} from the smallest to the largest quarter.")


def build_page(key, S, base_page, all_pooled):
    f, p, i, locs = S["f"], S["p"], S["i"], S["locs"]
    t = model_texts(key, S)
    tp, n_pred, prec, rec, f1 = pooled_row(S["poly"])
    sm = S["summary"]
    n_fields = len(f)
    regions = i.Region.value_counts()
    region_txt = ", ".join(f"<b>{r} ({n})</b>" for r, n in regions.sort_values(ascending=False).items())
    facts = (f'<div class="facts"><span><b>32</b> images</span><span><b>{n_fields:,}</b> reference fields</span><span><b>{n_pred:,}</b> predicted polygons</span>'
             f'<span>regions {region_txt}</span><span>field F1 <b>{f1:.3f}</b></span></div>')
    head = (f'<header style="display:grid;gap:14px">\n  <span class="eyebrow">{t["eyebrow"]}</span>\n  <h1>{t["h1"]}</h1>\n  <p class="lede">{t["lede"]}</p>\n  {facts}\n</header>\n')
    findings = "".join(f"    <li>{x}</li>\n" for x in t["findings"])
    pooled_rows = []
    for mkey, label in [("baseline", "DelAny V2"), ("ghana", "DelAny V2: Ghana-Trained"), ("combined", "DelAny V2: Ghana Viedia Trained"), ("ftp", "FTP"), ("ftw", "FTW")]:
        ptp, pn, pp, pr, pf = all_pooled[mkey]
        row = [label + (" (this page)" if mkey == key else ""), f"{pn:,}", f"{ptp:,}", f"{pp:.3f}", f"{pr:.3f}", f"{pf:.3f}"]
        pooled_rows.append((mkey == key, row))
    pooled = ('<div class="tw"><table><thead><tr><th>Model</th><th>Predicted polygons</th><th>Matched fields</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>'
              + "".join(f'<tr{" class=cur" if cur else ""}><td class="t">{r[0]}</td>' + "".join(f'<td class="n">{c}</td>' for c in r[1:]) + "</tr>" for cur, r in pooled_rows)
              + "</tbody></table></div>")
    model_section = (f'<section>\n  <h2>Model and input</h2>\n  <dl>{t["model_dl"]}</dl>\n'
                     f'  <h3>Full-image scores, all 32 images pooled</h3>\n  {pooled}\n'
                     f'  <p class="note">Counted over all {n_fields:,} reference fields and all predicted polygons at IoU ≥ 0.5. Mean IoU of the matched fields is {sm["mean_iou_of_matches"]:.3f}; '
                     f'{pct(sm["over_seg_fraction"])} of fields are split and {pct(sm["under_seg_fraction"])} of polygons merge fields (coverage rule of <code>evaluation.py</code>). '
                     f'mAP is not computed for this model; the tile-level scores on the Delineate Anything pages come from their training framework. '
                     f'Source: <code>{MODELS[key]["result_path"].replace("&lt;image&gt;", "…")}</code> scored with <code>evaluation.py</code>.</p>\n</section>\n')
    info = ('<section>\n  <h2>What information exists</h2>\n'
            '  <p class="col">The predicted polygons carry only their outline and area: this model gives no per-polygon confidence, and the satellite imagery behind the labels is not available here, '
            'so colour, greenness, texture and edge contrast are only given for the reference fields (measured earlier on the 1 m imagery), not for the predicted polygons. '
            'Everything else is joined in from the labels and the scores.</p>\n  <dl>\n'
            '    <dt>Labels (GPS survey)</dt><dd>Crop (<code>crop</code>: maize, rice, other, noncrop) and detailed crop (<code>crop1</code>: 24 values incl. fallow and unidentified), '
            'label confidence (high / medium / low / none), region (north, ej), survey date (15 Sep 2021 in the north, 15 Dec 2021 in ej, a few others).</dd>\n'
            '    <dt>Images (1 m RGB)</dt><dd>Measured per reference field: mean colour, greenness (VARI, from −1 to 1; higher is greener), texture (spread of brightness inside the field) '
            'and edge contrast (brightness difference between the field and a 3 m ring around it). These columns are the same for every model on this page.</dd>\n'
            '    <dt>Scores</dt><dd>Per image: precision, recall, F1, area omission, boundary F1, PoLiS. Per field: matched at IoU ≥ 0.5, best IoU, area covered, split, merged. '
            'PoLiS needs at least one polygon, so images without polygons have no value.</dd>\n'
            '    <dt>Not available</dt><dd>Prediction confidence, image properties of the predicted polygons, image dates, crop height or growth stage.</dd>\n  </dl>\n'
            '  <p class="note">How to read the outcome columns: <b>Found</b> = matched one-to-one with IoU ≥ 0.5 · <b>Area covered</b> = share of the field under any prediction · '
            '<b>Split</b> = covered by two or more predictions · <b>Merged</b> = its main prediction also covers another field.</p>\n</section>\n')

    # field size
    cuts = np.quantile(f.area, [0, .25, .5, .75, 1])
    size_rows, size_items = [], []
    for k in range(4):
        g = f[f.size_q == k]
        lab = f"Q{k + 1}: {cuts[k]:,.0f} m²–{cuts[k + 1]:,.0f} m²"
        size_rows.append(group_row(lab, g))
        size_items.append((lab, g.found.mean()))
    group_head = ["Field size", "Fields", "Found (IoU ≥ 0.5)", "Area covered", "Median best IoU", "Split", "Merged"]
    size = ('<section>\n  <h2>Field size</h2>\n  <div class="two">\n    <figure class="fig">' + bars(size_items)
            + '\n      <figcaption>Share of reference fields found, by size quarter (each quarter ≈ 384 fields).</figcaption></figure>\n'
            f'    <p class="col">{size_text(S)}</p>\n  </div>\n  ' + table(group_head, size_rows) + "\n</section>\n")

    # crop type
    crop_rows = [group_row(c, f[f.crop == c]) for c in CROP_ORDER]
    crop_items = sorted([(c, f[f.crop == c].found.mean()) for c in CROP_ORDER], key=lambda x: -x[1])
    crop_g = {c: f[f.crop == c] for c in CROP_ORDER}
    order = [c for c, _ in crop_items]
    swatches = re.search(r'<div class="swatches">.*?</div></div></div>\n', base_page, re.S).group(0).strip()
    crop = ('<section>\n  <h2>Crop type</h2>\n  <div class="two">\n    <figure class="fig">' + bars(crop_items)
            + '\n      <figcaption>Share of reference fields found, by crop group.</figcaption></figure>\n    ' + swatches + "\n  </div>\n"
            '  <p class="note">Swatches show each group\'s median field colour in the 1 m imagery. The model only learns "field", never the crop; differences come from how each crop looks and how big its fields are. '
            'Rice: almost all in the north. Other crops: 213 fields, mostly sorghum, yam, small mixed plots, groundnuts and cassava.</p>\n  '
            + table(["Crop group", *group_head[1:]], crop_rows) + "\n"
            f'  <p class="col">By crop, {order[0].lower()} is found most often ({pct(crop_items[0][1])}) and {order[-1].lower()} least ({pct(crop_items[-1][1])}). '
            f'Mean area covered is highest for {max(CROP_ORDER, key=lambda c: crop_g[c]["cov"].mean()).lower()} ({pct(max(g["cov"].mean() for g in crop_g.values()))}) '
            f'and lowest for {min(CROP_ORDER, key=lambda c: crop_g[c]["cov"].mean()).lower()} ({pct(min(g["cov"].mean() for g in crop_g.values()))}).</p>\n</section>\n')

    # label confidence and region
    conf_rows = [group_row(lab, f[f.conf == k]) for k, lab in CONF_ORDER]
    reg_label = {"ej": "ej (surveyed Dec 2021)", "north": "north (surveyed Sep 2021)"}
    reg_rows = [group_row(reg_label[r], f[f.img.map(lambda x: x[:7]).map(dict(zip(i.Image, i.Region))) == r]) for r in sorted(regions.index)]
    hi, unc = f[f.conf == "high"], f[f.conf.isin(["low", "none"])]
    n_found = int(f.found.sum())
    caution = "" if n_found >= 50 else f" With only {n_found} fields found in total, differences of a few percentage points are a handful of fields and should not be read as an effect."
    by_region = {r: f[f.img.str.slice(0, 7).map(dict(zip(i.Image, i.Region))) == r] for r in regions.index}
    cov_n, cov_e = by_region["north"]["cov"].mean(), by_region["ej"]["cov"].mean()
    region_note = ("The two regions perform alike" if abs(cov_n - cov_e) < 0.1 else
                  f"The regions differ: {pct(cov_n)} of a field's area is covered in the north against {pct(cov_e)} in ej, which follows how many images in each region get any polygon")
    label_sec = ('<section>\n  <h2>Label confidence and region</h2>\n  <div class="two">\n    '
                 + table(["Label confidence", *group_head[1:]], conf_rows) + "\n    " + table(["Region", *group_head[1:]], reg_rows) + "\n  </div>\n"
                 f'  <p class="col">Fields the surveyors were sure about have {pct(hi["cov"].mean())} of their area covered on average, against {pct(unc["cov"].mean())} for the uncertain ones (low or no confidence); '
                 f'the share found is {pct(hi.found.mean())} against {pct(unc.found.mean())}.{caution} {region_note}; region and survey date coincide, so their effects can\'t be separated.</p>\n</section>\n')

    # how fields look
    t_vari, fv, cv = quartile_table(f, "vari", "Greenness (VARI)", "VARI", lambda v: f"{v:+.2f}")
    t_tex, ft, ct = quartile_table(f, "tex", "Texture inside the field", "Texture", lambda v: f"{v:.0f}")
    t_edge, fe, ce = quartile_table(f, "edge", "Edge contrast with surroundings", "Contrast", lambda v: f"{v:.1f}")
    t_comp, fc, cc = quartile_table(f, "comp", "Shape (compactness)", "Compactness", lambda v: f"{v:.2f}")
    look = ('<section>\n  <h2>How fields look</h2>\n  <p class="col">Each table splits the 1,534 reference fields into four equal groups by one measured property.</p>\n  <div class="two">\n    '
            + "\n    ".join([t_vari, t_tex, t_edge, t_comp]) + "\n  </div>\n"
            f'  <p class="col"><b>Greenness</b>: from the least to the most green quarter, fields are covered {", ".join(cv)} of their area on average. '
            f'<b>Texture</b> (smoothest to roughest): {", ".join(ct)}. <b>Edge contrast</b> (weakest to sharpest): {", ".join(ce)}. <b>Compactness</b> (most irregular to most compact): {", ".join(cc)}. '
            f'The share found by quarter is {", ".join(fv)} (greenness), {", ".join(ft)} (texture), {", ".join(fe)} (edge contrast) and {", ".join(fc)} (compactness). '
            f'Brightness on its own makes little difference.</p>\n</section>\n')

    # predicted polygons
    shares = p.cat.value_counts(normalize=True)
    pred_items = [(label, float(shares.get(c, 0))) for c, label in PRED_CATS]
    several = p[p.cat == "Covers several fields"]
    big = (f" The predictions that merge several fields are the largest (median {several.area.median():,.0f} m²)." if len(several) else "")
    preds = ('<section>\n  <h2>What the predicted polygons are</h2>\n  <div class="two">\n    <figure class="fig">' + bars(pred_items)
             + f'\n      <figcaption>Share of the {n_pred:,} predicted polygons in each category.</figcaption></figure>\n'
             f'    <p class="col">Only {pct(shares.get("Matched field", 0))} of predictions are a clean match. {pct(shares.get("Piece of a field", 0))} are pieces of real fields or loose outlines inside them. '
             f'{pct(shares.get("On non-crop background", 0))} sit on the non-crop background; these are the real false detections according to the labels. '
             f'{pct(shares.get("Covers several fields", 0))} cover two or more fields.{big}</p>\n  </div>\n</section>\n')

    # image by image
    fa = f.assign(short=f.img.str.slice(0, 7))
    edge_med = fa.groupby("short").edge.median()
    ii = i.set_index("Image")
    props = [("Fields per hectare", ii["Fields / ha"]), ("Share of image that is non-crop", ii["Non-crop share"]), ("Median field size", ii["Median field (m²)"]),
             ("Image greenness (VARI)", ii["Mean VARI"]), ("Share of fields labelled unidentified or fallow", ii["Unident. + fallow"]),
             ("Median field edge contrast", edge_med.reindex(ii.index))]
    rho = sorted(((n, float(spearmanr(s, ii.F1).statistic)) for n, s in props), key=lambda x: -abs(x[1]))
    rho_rows = "".join(f'<tr><td class="t">{n}</td><td class="n">{r:+.2f}</td><td class="n"><span class="rho"><span style="width:{abs(r) * 100:.0f}%" class="{"pos" if r > 0 else "neg"}"></span></span></td></tr>'
                       for n, r in rho)
    full = {image_label(l)[:7]: image_label(l) for l in locs}
    ranked = i.sort_values(["F1", "Image"], ascending=[False, True])
    best = ", ".join(f"{r.Image} F1 {r.F1:.2f}" for r in ranked.head(3).itertuples() if r.F1 > 0)
    zero = int((i.F1 == 0).sum())
    nopoly = sorted(i[i.Predicted == 0].Image)
    img_text = (f'The best images ({best}). {zero} of 32 images score F1 = 0; '
                + (f"{len(nopoly)} of them have no polygon at all ({', '.join(nopoly)}). " if nopoly else "")
                + "With most images at zero the correlations on the left rest on the few images that score at all, so they say little; they are shown to match the other pages.")
    heads = ["Image", "Region", "Fields", "Predicted", "Fields / ha", "Median field", "Non-crop share", "Unident. + fallow", "Mean VARI", "Precision", "Recall", "F1", "Area omission", "Boundary F1 @ 5 m", "PoLiS (m)"]
    body = []
    for r in i.itertuples(index=False):
        c = dict(zip(i.columns, r))
        body.append('<tr><td class="t">' + full[c["Image"]] + '</td>' + "".join(f'<td class="n">{v}</td>' for v in [
            c["Region"], f'{c["Fields"]:.0f}', f'{c["Predicted"]:.0f}', f'{c["Fields / ha"]:.2f}', f'{c["Median field (m²)"]:,.0f} m²', pct(c["Non-crop share"]), pct(c["Unident. + fallow"]),
            f'{c["Mean VARI"]:+.2f}', f'{c["Precision"]:.2f}', f'{c["Recall"]:.2f}', f'{c["F1"]:.2f}', pct(c["Area omission"]), f'{c["Boundary F1 @ 5 m"]:.2f}',
            "–" if pd.isna(c["PoLiS (m)"]) else f'{c["PoLiS (m)"]:.1f}']) + "</tr>")
    imgsec = ('<section>\n  <h2>Image by image</h2>\n  <div class="two">\n    <div class="fig"><h3>What goes with a higher F1 across images</h3>\n'
              f'      <div class="tw"><table><thead><tr><th>Image property</th><th>Spearman ρ with F1</th><th></th></tr></thead><tbody>{rho_rows}</tbody></table></div>\n'
              '      <p class="note">32 images, so only |ρ| above about 0.35 is clearly more than noise. Blue: higher value, higher F1. Red: higher value, lower F1.</p></div>\n'
              f'    <p class="col">{img_text}</p>\n  </div>\n'
              '  <div class="tw"><table class="wide"><thead><tr>' + "".join(f"<th>{h}</th>" for h in heads) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>\n"
              '  <div class="xy" id="xy-img" data-title="images"></div>\n'
              '  <p class="note">Sorted by F1. Pick any two columns to plot them against each other; hover a point to see the image. Non-crop share = share of the image inside the non-crop background polygon. '
              'Area omission = share of field area not covered by any prediction. PoLiS is blank for images without polygons.</p>\n</section>\n')

    suggests = ('<section>\n  <h2>What this suggests</h2>\n  <ol class="find">\n' + "".join(f"    <li>{x}</li>\n" for x in t["suggests"]) + "  </ol>\n"
                f'  <p class="note">Data: per-image polygons in <code>{MODELS[key]["result_path"]}</code>, scored by <code>comparison_metrics.py</code> '
                '(same matching as <code>evaluation.py</code>); labels from <code>data/ground_truth/filtered_grids.gpkg</code>. The table of every polygon is at the end of this page. '
                'Outcome, split/merged and boundary definitions are listed at the top of <code>comparison_metrics.py</code>.</p>\n</section>\n')

    polys = ('<section id="polygons">\n  <h2>Every polygon</h2>\n  <p class="col">One row per reference field or predicted polygon. IDs are the <code>id</code> attribute in the label file '
             f'(<code>data/ground_truth/filtered_grids.gpkg</code>) and the <code>{MODELS[key]["id_column"]}</code> attribute in each image\'s prediction file '
             f'(<code>{MODELS[key]["result_path"]}</code>), so a row can be found in QGIS. Click a column header to sort.</p>\n'
             + re.search(r'  <div class="tabs" role="tablist">.*?</section>', base_page, re.S).group(0))
    return f'<template id="tpl-{key}">\n' + head + "\n<section>\n  <h2>Key findings</h2>\n  <ol class=\"find\">\n" + findings + "  </ol>\n</section>\n\n" \
        + model_section + info + "\n" + size + "\n" + crop + "\n" + label_sec + "\n" + look + "\n" + preds + "\n" + imgsec + "\n" + suggests + "\n" + polys + "\n</template>\n"


def image_label(loc):
    return f"area{loc['area']:03d}_{loc['grid']}"


# ---------------------------------------------------------------- page script
INIT_SCRIPT = r'''<script>
/* FTP and FTW pages: same tables and plots as the model pages above, driven by the embedded data of one model key. */
window.initModelPage = function (key) {
  const D = JSON.parse(document.getElementById("polydata-" + key).textContent);
  const COLS = {
    fields: [["Image","t"],["Label ID","t"],["Crop","t"],["Crop (detail)","t"],["Label conf.","t"],["Area (m²)","n"],["Outcome","t"],
             ["Best IoU","n"],["Covered","p"],["Colour","c"],["VARI","n"],["Texture","n"],["Edge contrast","n"],["Compactness","n"]],
    preds: [["Image","t"],["Polygon ID","n"],["Category","t"],["Area (m²)","n"],["In fields","p"],["In non-crop","p"],["Fields covered","n"],
            ["Colour","c"],["VARI","n"],["Texture","n"],["Edge contrast","n"],["Compactness","n"]]};
  const CAT = {fields: 6, preds: 2};
  const LEGEND = {
    fields: "Outcome: Found = matched one-to-one at IoU ≥ 0.5 · Missed = less than 20% of the field under any prediction · Split = covered by two or more predictions · Merged = its main prediction also covers another field · Poor outline = covered, but no prediction fits it well. Covered = share of the field's area under any prediction. Colour = mean colour of the field in the image.",
    preds: "Category: Matched field = matches one reference field at IoU ≥ 0.5 · Piece of a field = mostly inside reference fields but no match · On non-crop background = at least half on the non-crop polygon · Covers several fields = covers two or more reference fields · Mixed = straddles fields and background. In fields / In non-crop = share of the polygon's area. Colour, VARI, texture and edge contrast are not available for this model's polygons."};
  const PAGE = 50;
  const polyXY = makeXY(document.getElementById("xy-poly"), r => `${r[0]} · ID ${r[1]}`);
  const IMG = JSON.parse(document.getElementById("imgdata-" + key).textContent);
  makeXY(document.getElementById("xy-img"), r => r[0]).update(IMG.cols, IMG.rows, "img", {x: "Fields / ha", y: "F1", c: "Region"});
  const st = {tab: "fields", image: "", cat: "", crop: "", sort: null, dir: 1, page: 0};
  const $ = id => document.getElementById(id);
  const uniq = (rows, i) => [...new Set(rows.map(r => r[i]))].sort();
  const opts = (sel, vals, all) => { sel.innerHTML = `<option value="">${all}</option>` + vals.map(v => `<option>${v}</option>`).join(""); };
  $("n-fields").textContent = D.fields.length.toLocaleString(); $("n-preds").textContent = D.preds.length.toLocaleString();
  function setup() {
    const rows = D[st.tab];
    opts($("f-image"), uniq(rows, 0), "All images");
    opts($("f-cat"), uniq(rows, CAT[st.tab]), st.tab === "fields" ? "All outcomes" : "All categories");
    $("f-cat-label").textContent = st.tab === "fields" ? "Outcome" : "Category";
    $("f-crop-wrap").hidden = st.tab !== "fields";
    if (st.tab === "fields") opts($("f-crop"), uniq(rows, 2), "All crops");
    $("f-image").value = st.image; $("f-cat").value = ""; st.cat = ""; st.crop = ""; st.sort = null; st.page = 0;
    $("legend").textContent = LEGEND[st.tab];
    $("tab-fields").setAttribute("aria-selected", st.tab === "fields"); $("tab-preds").setAttribute("aria-selected", st.tab === "preds");
    render();
  }
  function fmt(v, kind) {
    if (v === null || v === "") return "–";
    if (kind === "p") return Math.round(v * 100) + "%";
    if (kind === "n" && typeof v === "number") return v.toLocaleString(undefined, {maximumFractionDigits: 2});
    return v;
  }
  function render() {
    const cols = COLS[st.tab];
    let rows = D[st.tab].filter(r => (!st.image || r[0] === st.image) && (!st.cat || r[CAT[st.tab]] === st.cat) && (!st.crop || st.tab !== "fields" || r[2] === st.crop));
    if (st.sort !== null) rows = rows.slice().sort((a, b) => {
      const x = a[st.sort], y = b[st.sort];
      if (x === null) return 1; if (y === null) return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * st.dir; });
    const pages = Math.max(1, Math.ceil(rows.length / PAGE)); st.page = Math.min(st.page, pages - 1);
    $("poly").tHead.innerHTML = "<tr>" + cols.map(([h, k], i) =>
      `<th class="${k === "t" || k === "c" ? "" : "num"}"><button data-i="${i}" aria-label="Sort by ${h}">${h}${st.sort === i ? (st.dir > 0 ? " ▲" : " ▼") : ""}</button></th>`).join("") + "</tr>";
    $("poly").tBodies[0].innerHTML = rows.slice(st.page * PAGE, st.page * PAGE + PAGE).map(r => "<tr>" + cols.map(([h, k], i) =>
      k === "c" ? `<td><span class="dot" style="background:${r[i] || "transparent"}" title="${r[i]}"></span></td>`
                : `<td class="${k === "t" ? "t" : "n"}">${fmt(r[i], k)}</td>`).join("") + "</tr>").join("");
    $("shown").textContent = rows.length.toLocaleString() + " shown";
    polyXY.update(cols, rows, st.tab, st.tab === "fields" ? {x: "Area (m²)", y: "Best IoU", c: "Outcome", lx: true}
                                                         : {x: "Area (m²)", y: "In non-crop", c: "Category", lx: true});
    $("pageinfo").textContent = `Page ${st.page + 1} of ${pages}`;
    $("prev").disabled = st.page === 0; $("next").disabled = st.page >= pages - 1;
  }
  $("poly").tHead.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
    const i = +b.dataset.i; st.dir = st.sort === i ? -st.dir : 1; st.sort = i; st.page = 0; render(); });
  $("f-image").addEventListener("change", e => { st.image = e.target.value; st.page = 0; render(); });
  $("f-cat").addEventListener("change", e => { st.cat = e.target.value; st.page = 0; render(); });
  $("f-crop").addEventListener("change", e => { st.crop = e.target.value; st.page = 0; render(); });
  $("prev").addEventListener("click", () => { st.page--; render(); });
  $("next").addEventListener("click", () => { st.page++; render(); });
  $("tab-fields").addEventListener("click", () => { st.tab = "fields"; setup(); });
  $("tab-preds").addEventListener("click", () => { st.tab = "preds"; setup(); });
  setup();
};
window.INIT["ftp"] = () => window.initModelPage("ftp");
window.INIT["ftw"] = () => window.initModelPage("ftw");
</script>
'''


# ---------------------------------------------------------------- patching
def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def put_block(html, name, content, before):
    """Replace the marker block `name`, or insert it right before `before` the first time."""
    begin, end = MARK.format(name=name, edge="begin"), MARK.format(name=name, edge="end")
    block = f"{begin}\n{content}{end}\n"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n", re.S)
    if pattern.search(html):
        return pattern.sub(lambda m: block, html)
    assert html.count(before) == 1, f"anchor for {name} not unique: {before!r}"
    return html.replace(before, block + before)


def main():
    html = read_html()
    labels = gpd.read_file(ROOT / "data/ground_truth/filtered_grids.gpkg")
    locs = cm.load_locations()
    base = {"img": embedded(html, "imgdata-baseline"), "poly": embedded(html, "polydata-baseline")}
    base_page = re.search(r'<template id="tpl-baseline">(.*?)</template>', html, re.S).group(1)
    pooled = {k: pooled_row(embedded(html, f"polydata-{k}")) for k in ("baseline", "ghana", "combined")}
    results = {}
    for key, m in MODELS.items():
        img, poly, flags = cm.compute_model(m["results"], base, labels, locs, m["id_column"])
        summary = json.loads((m["results"] / "scores_test_summary.json").read_text())
        results[key] = dict(img=img, poly=poly, flags=flags, summary=summary)
        pooled[key] = pooled_row(poly)
        assert abs(pooled[key][4] - summary["f1"]) < 1e-9 and pooled[key][0] == summary["tp"], f"{key}: pooled scores differ from scores_test_summary.json"
    templates, data = "", ""
    for key in MODELS:
        r = results[key]
        S = stats_for(key, r["img"], r["poly"], r["flags"], r["summary"], locs)
        S["poly"] = r["poly"]
        templates += build_page(key, S, base_page, pooled)
        data += (f'<script type="application/json" id="imgdata-{key}">{compact(r["img"])}</script>\n'
                 f'<script type="application/json" id="polydata-{key}">{compact(r["poly"])}</script>\n')
    html = put_block(html, "templates", templates, "<script>\nwindow.makeXY = ")
    html = put_block(html, "data", data, "<script>window.INIT = {};</script>\n")
    html = put_block(html, "init", INIT_SCRIPT, '<script>\nwindow.INIT["compare"]')
    write_html(html)


if __name__ == "__main__":
    main()

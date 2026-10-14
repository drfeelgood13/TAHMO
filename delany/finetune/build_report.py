"""Builds the Ghana test-set diagnostics page from finetune/results/<run>/analysis/* (made by analyze_predictions.py).

Usage: python finetune/build_report.py finetune/results/combined_ghana/analysis out.html \
           --model combined_1m --model-desc "fine-tuned on Ghana + VIEDIA at 1 m" --title "Ghana Test-Set Diagnostics"
"""
import argparse
import html
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("analysis"); ap.add_argument("out")
ap.add_argument("--model", default="combined_1m")
ap.add_argument("--model-desc", default="fine-tuned on Ghana + VIEDIA at 1 m")
ap.add_argument("--title", default="Ghana Test-Set Diagnostics")
ap.add_argument("--tile-name", help="this model's row name in the tile-score table (default: --model)")
ap.add_argument("--tile-scores", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tile_scores_ghana_1m.csv"),
                help="tile-level mAP per model on the Ghana 1 m test tiles (from the test job logs)")
args = ap.parse_args()
TILE_NAME = args.tile_name or args.model
A, OUT = args.analysis, args.out
RES = os.path.relpath(os.path.dirname(os.path.abspath(A.rstrip("/"))), os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # e.g. finetune/results/ghana_1m
RES_SHORT = RES.replace("finetune/", "")
S = json.load(open(f"{A}/summary.json"))
I = pd.read_csv(f"{A}/images.csv")
F = pd.read_csv(f"{A}/fields.csv")
P = pd.read_csv(f"{A}/predictions.csv")

pct = lambda v: f"{100 * v:.0f}%"
num = lambda v, d=0: f"{v:,.{d}f}"
esc = html.escape


def bars(rows, value_key, label_key, fmt=pct, maxv=None, note_key=None, lw=150):
    """Horizontal bar chart as inline SVG, one series, labels from the data (lw = width of the label column)."""
    maxv = maxv or max(r[value_key] for r in rows) or 1
    bw, rh = 260, 26
    h = rh * len(rows) + 6
    out = [f'<svg class="bars" viewBox="0 0 {lw + bw + 70} {h}" role="img" preserveAspectRatio="xMinYMin meet">']
    for k, r in enumerate(rows):
        y = 4 + k * rh
        w = max(bw * r[value_key] / maxv, 1.5)
        out.append(f'<text x="{lw - 10}" y="{y + 15}" text-anchor="end" class="bl">{esc(str(r[label_key]))}</text>')
        out.append(f'<rect x="{lw}" y="{y + 4}" width="{bw}" height="14" rx="2" class="track"/>')
        out.append(f'<rect x="{lw}" y="{y + 4}" width="{w:.1f}" height="14" rx="2" class="bar"><title>{esc(str(r[label_key]))}: {fmt(r[value_key])}</title></rect>')
        out.append(f'<text x="{lw + w + 6:.1f}" y="{y + 15}" class="bv">{fmt(r[value_key])}</text>')
    out.append("</svg>")
    return "".join(out)


def table(df, cols, heads, fmts, cls=""):
    th = "".join(f"<th>{h}</th>" for h in heads)
    trs = []
    for _, r in df.iterrows():
        tds = "".join(f'<td class="{"t" if i == 0 else "n"}">{f(r[c])}</td>' for i, (c, f) in enumerate(zip(cols, fmts)))
        trs.append(f"<tr>{tds}</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


outcome_cols = ["n", "found", "covered", "best_iou", "split", "merged"]
outcome_heads = ["Fields", "Found (IoU ≥ 0.5)", "Area covered", "Median best IoU", "Split", "Merged"]
outcome_fmts = [lambda v: num(v), pct, pct, lambda v: f"{v:.2f}", pct, pct]


def outcome_table(key, label, rename=None):
    df = pd.DataFrame(S[key])
    first = df.columns[0]
    if rename:
        df[first] = df[first].map(lambda v: rename.get(str(v), str(v)))
    return table(df, [first] + outcome_cols, [label] + outcome_heads, [str] + outcome_fmts)


def quartile_labels(key, unit_fmt):
    df = pd.DataFrame(S[key]); first = df.columns[0]
    labs = []
    for i, iv in enumerate(df[first]):
        lo, hi = [float(x) for x in str(iv).strip("(]").split(",")]
        labs.append(f"Q{i + 1}: {unit_fmt(lo)}–{unit_fmt(hi)}")
    df[first] = labs
    return df, first


# ---------- numbers used in the text
n_f, n_p = S["fields"], S["predictions"]
found = F.matched.mean(); covered = F.covered_share.mean(); split = F.split.mean(); merged = F.merged.mean()
size_df, size_col = quartile_labels("by_size", lambda v: f"{max(v, 20):,.0f} m²")
crop = pd.DataFrame(S["by_crop_group"]).set_index("crop_group")
conf = pd.DataFrame(S["by_label_confidence"]).set_index("label_confidence")
kinds = pd.DataFrame(S["prediction_kinds"]).rename(columns={"index": "kind"})
kinds = kinds.rename(columns={kinds.columns[0]: "kind"})
app = pd.DataFrame(S["appearance_by_crop_group"]).set_index("crop_group")
corr = pd.DataFrame(S["image_correlations_with_f1"]); corr = corr.rename(columns={corr.columns[0]: "factor"})
vari_df, vari_col = quartile_labels("by_vari", lambda v: f"{v:+.2f}")
tex_df, tex_col = quartile_labels("by_texture", lambda v: f"{v:.0f}")
edge_df, edge_col = quartile_labels("by_edge_contrast", lambda v: f"{v:.1f}")
bri_df, bri_col = quartile_labels("by_brightness", lambda v: f"{v:.0f}")
comp_df, comp_col = quartile_labels("by_compactness", lambda v: f"{v:.2f}")
kind_n = kinds.set_index("kind").n
reg = pd.DataFrame(S["by_region"]).set_index("region")

crop_names = {"maize": "Maize", "rice": "Rice", "other crops": "Other crops", "fallow": "Fallow", "unidentified": "Unidentified"}
crop_order = ["rice", "maize", "fallow", "unidentified", "other crops"]

# ---------- swatches: median field colour per crop group (real image colour)
sw = []
for c in crop_order:
    r = app.loc[c]
    sw.append(f'''<div class="sw"><span class="chip" style="background:rgb({r.r:.0f},{r.g:.0f},{r.b:.0f})"></span>
<div><b>{crop_names[c]}</b><span class="m">RGB {r.r:.0f} · {r.g:.0f} · {r.b:.0f}</span>
<span class="m">VARI {r.vari:+.2f} · texture {r.texture:.1f} · edge {r.edge_contrast:.1f}</span>
<span class="m">median {r.area_m2:,.0f} m²</span></div></div>''')

# ---------- per-image table
I2 = I.sort_values("f1", ascending=False).copy()
I2["name"] = I2.image.str.replace("GHANA_", "").str.replace("_1m", "")
img_table = table(I2, ["name", "region", "n_fields", "n_pred", "fields_per_ha", "median_field_m2", "noncrop_share",
                       "share_unidentified_fallow", "vari", "precision", "recall", "f1", "area_omission", "boundary_F1_5m", "polis_mean_m"],
                  ["Image", "Region", "Fields", "Predicted", "Fields / ha", "Median field", "Non-crop share", "Unident. + fallow",
                   "Mean VARI", "Precision", "Recall", "F1", "Area omission", "Boundary F1 @ 5 m", "PoLiS (m)"],
                  [str, str, num, num, lambda v: f"{v:.2f}", lambda v: f"{v:,.0f} m²", pct, pct, lambda v: f"{v:+.2f}",
                   lambda v: f"{v:.2f}", lambda v: f"{v:.2f}", lambda v: f"{v:.2f}", pct, lambda v: f"{v:.2f}", lambda v: f"{v:.1f}"], "wide")

img_cols = [["Image", "t"], ["Region", "t"], ["Fields", "n"], ["Predicted", "n"], ["Fields / ha", "n"], ["Median field (m²)", "n"],
            ["Non-crop share", "p"], ["Unident. + fallow", "p"], ["Mean VARI", "n"], ["Precision", "n"], ["Recall", "n"], ["F1", "n"],
            ["Area omission", "p"], ["Boundary F1 @ 5 m", "n"], ["PoLiS (m)", "n"]]
img_keys = ["name", "region", "n_fields", "n_pred", "fields_per_ha", "median_field_m2", "noncrop_share", "share_unidentified_fallow",
            "vari", "precision", "recall", "f1", "area_omission", "boundary_F1_5m", "polis_mean_m"]
img_json = json.dumps({"cols": img_cols, "rows": [[(r[k].split("_")[0] if k == "name" else (round(float(r[k]), 4) if isinstance(r[k], (int, float)) else r[k]))
                                                   for k in img_keys] for _, r in I2.iterrows()]}, separators=(",", ":")).replace("</", "<\\/")

corr_names = {"fields_per_ha": "Fields per hectare", "median_field_m2": "Median field size", "noncrop_share": "Share of image that is non-crop",
              "brightness": "Image brightness", "vari": "Image greenness (VARI)", "field_edge_contrast": "Median field edge contrast",
              "share_unidentified_fallow": "Share of fields labelled unidentified or fallow"}
corr["label"] = corr.factor.map(corr_names)
corr = corr.reindex(corr.spearman_rho.abs().sort_values(ascending=False).index)
corr_rows = "".join(f'<tr><td class="t">{esc(r.label)}</td><td class="n">{r.spearman_rho:+.2f}</td>'
                    f'<td class="n"><span class="rho"><span style="width:{abs(r.spearman_rho) * 100:.0f}%" class="{"pos" if r.spearman_rho > 0 else "neg"}"></span></span></td></tr>'
                    for r in corr.itertuples())

kind_text = {"matched field": "Matches one reference field (IoU ≥ 0.5)",
             "piece of a field / poor outline": "Mostly inside reference fields, but no match: a piece of a split field or a loose outline",
             "mostly in non-crop area": "At least half inside the non-crop background polygon",
             "covers several fields": "Covers two or more reference fields (merged)",
             "mixed / boundary": "Straddles fields and background"}
kind_rows = [{"k": kind_text[k], "v": kinds.set_index("kind").share[k]} for k in
             ["matched field", "piece of a field / poor outline", "mostly in non-crop area", "covers several fields", "mixed / boundary"]]

size_rows = [{"l": r[size_col], "v": r.found} for _, r in size_df.iterrows()]
crop_rows = [{"l": crop_names[c], "v": crop.found[c]} for c in crop_order]
small_found = size_df.found.iloc[0]; big_found = size_df.found.iloc[3]; big_split = size_df.split.iloc[3]; small_merged = size_df.merged.iloc[0]

kind_short = {"matched field": "Matched field", "piece of a field / poor outline": "Piece of a field",
              "mostly in non-crop area": "On non-crop background", "covers several fields": "Covers several fields",
              "mixed / boundary": "Mixed / boundary"}

# ---------- model confidence (present when the run was made with save_detections: true and confidence_cutoff.py ran)
HAS_CONF = "confidence" in P.columns and os.path.exists(f"{A}/cutoff_sweep.csv")
if HAS_CONF:
    SW = pd.read_csv(f"{A}/cutoff_sweep.csv")
    CK = pd.read_csv(f"{A}/confidence_by_kind.csv").set_index("kind")
    best_cut = SW.loc[SW.f1.idxmax()]


def sweep_chart(sw):
    """P / R / F1 against the confidence cut-off, inline SVG with a legend and direct end labels."""
    W, H, L, R, T, B = 560, 250, 44, 86, 12, 34
    xs = lambda v: L + (v - sw.cutoff.min()) / (sw.cutoff.max() - sw.cutoff.min()) * (W - L - R)
    ys = lambda v: T + (1 - v / 0.6) * (H - T - B)
    o = [f'<svg class="sweep" viewBox="0 0 {W} {H}" role="img" aria-label="Precision, recall and F1 against the confidence cut-off">']
    for g in (0, 0.2, 0.4, 0.6):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(g):.1f}" y2="{ys(g):.1f}" class="grid"/><text x="{L - 8}" y="{ys(g) + 4:.1f}" text-anchor="end" class="ax">{g:.1f}</text>')
    for c in sw.cutoff:
        if round(c * 100) % 10 == 0 or c == sw.cutoff.min():
            o.append(f'<text x="{xs(c):.1f}" y="{H - B + 18}" text-anchor="middle" class="ax">{c:.2f}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 2}" text-anchor="middle" class="ax">confidence cut-off</text>')
    ends = []
    for key, cls, name in (("precision", "s1", "Precision"), ("recall", "s2", "Recall"), ("f1", "s3", "F1")):
        pts = " ".join(f"{xs(c):.1f},{ys(v):.1f}" for c, v in zip(sw.cutoff, sw[key]))
        o.append(f'<polyline points="{pts}" class="ln {cls}"/>')
        for c, v in zip(sw.cutoff, sw[key]):
            o.append(f'<circle cx="{xs(c):.1f}" cy="{ys(v):.1f}" r="4" class="pt {cls}"><title>{name} at {c:.2f}: {v:.3f}</title></circle>')
        ends.append([ys(sw[key].iloc[-1]), name, cls])
    ends.sort()
    for k in range(1, len(ends)):  # keep end labels apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 13)
    for y, name, cls in ends:
        o.append(f'<text x="{W - R + 8}" y="{y + 4:.1f}" class="lab {cls}t">{name}</text>')
    o.append(f'<line x1="{xs(0.15):.1f}" x2="{xs(0.15):.1f}" y1="{T}" y2="{H - B}" class="cur"/><text x="{xs(0.15) + 4:.1f}" y="{T + 10}" class="ax">current</text>')
    o.append("</svg>")
    return "".join(o)


# ---------- sentences that depend on the model's numbers
nc_share = kinds.set_index("kind").share["mostly in non-crop area"]
conf_ratio = conf.found["high"] / ((conf.found["low"] + conf.found["none"]) / 2)
others_vari = vari_df.found.iloc[:3]
best_crop, worst_crop = crop.found.idxmax(), crop.found.idxmin()
crop_finding = (f"<li><b>Rice is easiest, mixed small crops are hardest.</b> Rice fields are found {pct(crop.found['rice'])} of the time: they are greener, smoother and have the "
                f"sharpest edges. \"Other crops\" (sorghum, groundnut, yam, vegetables…) are found only {pct(crop.found['other crops'])} of the time: they are the smallest group with the weakest edges.</li>"
                if (best_crop, worst_crop) == ("rice", "other crops") else
                f"<li><b>Crop groups differ.</b> {crop_names[best_crop]} fields are found most often ({pct(crop.found[best_crop])}), "
                f"{crop_names[worst_crop].lower()} least often ({pct(crop.found[worst_crop])}).</li>")
Ib = I.sort_values("f1", ascending=False).assign(name=lambda d: d.image.str.replace("GHANA_", "").str.replace("_1m", ""))
best3, worst4 = Ib.head(3), Ib.tail(4).iloc[::-1]


def why(r):
    reasons = []
    if r.median_field_m2 < 1000: reasons.append(f"very small fields (median {r.median_field_m2:,.0f} m²)")
    if r.n_fields <= 15 and r.noncrop_share >= 0.5: reasons.append(f"only {r.n_fields} fields in a mostly non-crop scene")
    if r.share_unidentified_fallow >= 0.6: reasons.append(f"{pct(r.share_unidentified_fallow)} of fields unidentified or fallow")
    return "; ".join(reasons) or f"{r.n_pred} predictions for {r.n_fields} fields"


if HAS_CONF:
    ck = lambda k, c: CK.loc[k, c]
    CONF_INTRO = ("The predicted polygons carry their outline, area and the model's confidence. Confidence comes from a re-run with "
                  "<code>save_detections: true</code> that reproduced the original results exactly. Everything else below is joined in from the labels, the images and the scores.")
    CONF_DT = ("<dt>Model confidence</dt><dd>Per predicted polygon: the confidence of the raw detection that matches it best (0.15 to 1; the pipeline drops detections below 0.15), "
               "plus the highest confidence among the detections it was built from.</dd>"
               "<dt>Not available</dt><dd>Image dates, crop height or growth stage.</dd>")
    CONF_FINDING = (f"<li><b>A higher confidence cut-off does not help.</b> False detections on the background have slightly lower confidence (median "
                    f"{ck('mostly in non-crop area', '50%'):.2f} against {ck('matched field', '50%'):.2f} for matched fields), but the groups overlap so much that raising "
                    f"the cut-off removes good fields almost as fast as bad ones. F1 is highest at the current cut-off of {best_cut.cutoff:.2f}.</li>")
    CONF_SUGGEST = (f"<li><b>Keep the confidence cut-off at {best_cut.cutoff:.2f}, or test a lower one.</b> Raising it trades recall for precision at a loss in F1 "
                    f"(see Model confidence). Since F1 still falls as the cut-off rises from {best_cut.cutoff:.2f}, a cut-off of 0.10 is worth one run on the validation images.</li>")
    kind_order = ["matched field", "covers several fields", "mixed / boundary", "piece of a field / poor outline", "mostly in non-crop area"]
    ck_rows = "".join(f'<tr><td class="t">{kind_short[k]}</td><td class="n">{num(CK.loc[k, "count"])}</td><td class="n">{CK.loc[k, "50%"]:.2f}</td>'
                      f'<td class="n">{CK.loc[k, "25%"]:.2f}–{CK.loc[k, "75%"]:.2f}</td><td class="n">{pct(CK.loc[k, "share >= 0.3"])}</td></tr>' for k in kind_order)
    sw_rows = "".join(f'<tr{" class=cur" if abs(r.cutoff - best_cut.cutoff) < 1e-9 else ""}><td class="n">{r.cutoff:.2f}</td><td class="n">{num(r.predictions)}</td>'
                      f'<td class="n">{num(r.matched)}</td><td class="n">{r.precision:.3f}</td><td class="n">{r.recall:.3f}</td><td class="n">{r.f1:.3f}</td>'
                      f'<td class="n">{pct(r.split_share)}</td></tr>' for r in SW.itertuples())
    CONF_SECTION = f"""<section>
  <h2>Model confidence</h2>
  <p class="col">For every field it draws, the model gives a confidence between 0 and 1. The pipeline keeps everything at or above 0.15. If false detections had clearly lower
  confidence than correct ones, a higher cut-off would remove them cheaply.</p>
  <div class="two">
    <div class="fig"><h3>Confidence by kind of prediction</h3>
      <div class="tw"><table><thead><tr><th>Prediction</th><th>Count</th><th>Median</th><th>Middle half</th><th>Share ≥ 0.3</th></tr></thead><tbody>{ck_rows}</tbody></table></div>
      <p class="note">Background detections score lowest and matched fields highest, but the middle halves overlap almost completely.</p></div>
    <figure class="fig">{sweep_chart(SW)}
      <div class="legend"><span><i style="background:var(--s1)"></i>Precision</span><span><i style="background:var(--s2)"></i>Recall</span><span><i style="background:var(--s3)"></i>F1</span></div>
      <figcaption>Precision, recall and F1 when only polygons at or above a cut-off are kept. Recall falls faster than precision rises, so F1 drops.</figcaption></figure>
  </div>
  <div class="tw"><table><thead><tr><th>Cut-off</th><th>Predictions kept</th><th>Matched</th><th>Precision</th><th>Recall</th><th>F1</th><th>Fields split</th></tr></thead><tbody>{sw_rows}</tbody></table></div>
  <p class="note">The sweep filters the finished map. A re-run with a higher <code>minimal_confidence</code> would also change how pieces merge, so treat it as a close guide rather than an exact result.
  Per-polygon confidence: <code>{RES}/*.confidence.gpkg</code> and <code>confidence_polygons.csv</code>.</p>
</section>"""
else:
    CONF_INTRO = ("The predicted polygons themselves carry no attributes beyond their outline and area: this run did not record model confidence. "
                  "Everything below is joined in from the labels, the images and the scores.")
    CONF_DT = ("<dt>Not available</dt><dd>Prediction confidence for this run (it can be added by re-running with <code>save_detections: true</code>, about 30 minutes), "
               "image dates, crop height or growth stage.</dd>")
    CONF_FINDING = CONF_SECTION = ""
    CONF_SUGGEST = ("<li><b>Record confidence next time.</b> Re-running with <code>save_detections: true</code> would show whether false detections on the background "
                    "have lower confidence, which would make a higher confidence cut-off an easy precision gain.</li>")

TS = pd.read_csv(args.tile_scores) if os.path.exists(args.tile_scores) else None
if TS is not None:
    ts_rows = "".join(
        f'<tr{" class=cur" if r.model == TILE_NAME else ""}><td class="t">{esc(r.model)}{" (this page)" if r.model == TILE_NAME else ""}</td>'
        f'<td class="n">{r.mask_p:.3f}</td><td class="n">{r.mask_r:.3f}</td><td class="n">{r.mask_f1:.2f}</td><td class="n">{r.mask_map50:.3f}</td>'
        f'<td class="n">{r.mask_map50_95:.3f}</td><td class="n">{r.box_map50:.3f}</td><td class="n">{r.box_map50_95:.3f}</td></tr>' for r in TS.itertuples())
    ts_bar = bars([{"l": r.model, "v": r.mask_map50} for r in TS.itertuples()], "v", "l", fmt=lambda v: f"{v:.3f}", maxv=max(0.3, TS.mask_map50.max()),
                  lw=max(150, 10 + 7 * max(len(m) for m in TS.model)))
    me, v2 = TS.set_index("model").loc[TILE_NAME], TS.iloc[0]  # the baseline is the first row
    if TILE_NAME == v2.model:
        others = TS[TS.model != v2.model]
        gain_text = (f"This is the starting point: v2 without fine-tuning reaches mask mAP50 {v2.mask_map50:.3f}. Fine-tuning raises it to "
                     + ", ".join(f"{r.mask_map50:.3f} ({esc(r.model)})" for r in others.itertuples()) + ".")
        gain_text = gain_text[::-1].replace(" ,", " dna ", 1)[::-1] if len(others) > 1 else gain_text  # "a, b and c"
    else:
        gain_text = (f"{esc(args.model)} raises mask mAP50 from {v2.mask_map50:.3f} (v2 without fine-tuning) to {me.mask_map50:.3f}, "
                     f"a {100 * (me.mask_map50 / v2.mask_map50 - 1):.0f}% gain.")
    MAP_SECTION = f"""<section>
  <h2>Tile-level accuracy (mAP)</h2>
  <div class="two">
    <figure class="fig">{ts_bar}<figcaption>Mask mAP50 on the 130 Ghana 1 m test tiles (32 images), per model.</figcaption></figure>
    <p class="col">These scores judge the model on its own, tile by tile, before the delineation pipeline joins tiles into maps. <b>mAP50</b> summarises
    precision and recall over every confidence cut-off, counting a field as found at IoU ≥ 0.5; <b>mAP50-95</b> averages stricter overlap thresholds up to 0.95,
    so it rewards tight outlines. Precision, recall and F1 here are at the cut-off where F1 is highest for that model. {gain_text}</p>
  </div>
  <div class="tw"><table><thead><tr><th>Model</th><th>Mask precision</th><th>Mask recall</th><th>Mask F1</th><th>Mask mAP50</th><th>Mask mAP50-95</th>
  <th>Box mAP50</th><th>Box mAP50-95</th></tr></thead><tbody>{ts_rows}</tbody></table></div>
  <p class="note">Tile scores are not comparable with the full-image F1 used elsewhere on this page: tiles are scored one by one at the model's best cut-off,
  full images after merging, at the fixed cut-off of 0.15. Source: Ultralytics validation in the test jobs (<code>finetune/runs/delany-test_*.log</code>).</p>
</section>"""
else:
    MAP_SECTION = ""

sz = size_df
big_merged = sz.merged.iloc[3]
big_fate = (f"the largest fields are mostly broken into pieces ({pct(big_split)} split)" if big_split >= big_merged
            else f"the largest fields are more often merged with neighbours ({pct(big_merged)}) than split ({pct(big_split)})")
best_q = int(sz.found.values.argmax())
size_text = (f"Fields under about 740 m² are found only {pct(sz.found.iloc[0])} of the time; {pct(sz.merged.iloc[0])} of them end up inside a larger prediction. "
             f"Fields are found most often in the {['smallest', 'second', 'third', 'largest'][best_q]} size quarter ({pct(sz.found.iloc[best_q])}). "
             f"In the largest quarter, {pct(sz.split.iloc[3])} of fields are cut into several pieces and {pct(sz.merged.iloc[3])} are merged with neighbours. "
             "Both effects point at the delineation settings (minimum area, merge thresholds) as much as at the model.")
vq, tq, eq, cq, bq = (d.reset_index(drop=True) for d in (vari_df, tex_df, edge_df, comp_df, bri_df))
if vq.found.iloc[3] == vq.found.min():
    vari_text = (f"<b>The greenest quarter is found least often</b> ({pct(vq.found.iloc[3])}, against {pct(vq.found.iloc[:3].min())}–{pct(vq.found.iloc[:3].max())} "
                 "for the others). These are probably tall or dense crops, or fields next to trees and bush, where the boundary is hidden by vegetation.")
else:
    vari_text = ("<b>Greenness</b>: from the least to the most green quarter, fields are found " + ", ".join(pct(v) for v in vq.found)
                 + f" of the time; the {['least green', 'second', 'third', 'greenest'][int(vq.found.values.argmin())]} quarter is found least often.")
tex_text = ("<b>Rough, patchy fields</b> (high texture) are found less often" + (" and split more" if tq.split.iloc[3] > tq.split.iloc[0] else "")
            + f" ({pct(tq.found.iloc[3])} against {pct(tq.found.iloc[0])} for the smoothest)." if tq.found.iloc[3] < tq.found.iloc[0]
            else f"<b>Texture</b> makes little consistent difference ({pct(tq.found.iloc[0])} found for the smoothest quarter, {pct(tq.found.iloc[3])} for the roughest).")
edge_text = ("<b>Sharp edges help</b>: fields that stand out from their surroundings are found more often" + (" and merged less" if eq.merged.iloc[3] < eq.merged.iloc[0] else "")
             + f" ({pct(eq.found.iloc[3])} against {pct(eq.found.iloc[0])})." if eq.found.iloc[3] > eq.found.iloc[0]
             else f"<b>Edge contrast</b> makes little consistent difference ({pct(eq.found.iloc[0])} against {pct(eq.found.iloc[3])}).")
comp_text = (f"<b>Irregular, elongated fields</b> (low compactness) are found least often ({pct(cq.found.iloc[0])})." if cq.found.iloc[0] == cq.found.min()
             else "<b>Shape</b>: " + ", ".join(pct(v) for v in cq.found) + " found, from the least to the most compact quarter.")
bri_text = ("Brightness on its own makes little difference." if bq.found.max() - bq.found.min() < 0.08
            else "Brightness matters too: " + ", ".join(pct(v) for v in bq.found) + " found, from the darkest to the brightest quarter.")
piece_share = kinds.set_index("kind").share["piece of a field / poor outline"]
piece_word = "Almost half" if 0.4 <= piece_share < 0.5 else pct(piece_share)

image_text = ("The best images (" + ", ".join(f"{r.name.split('_')[0]} F1 {r.f1:.2f}" for r in best3.itertuples())
              + ") tend to have many regular fields and little background. The worst: "
              + "; ".join(f"<b>{r.name.split('_')[0]}</b> (F1 {r.f1:.2f}): {why(r)}" for r in worst4.itertuples()) + ".")

page = f"""<title>{esc(args.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,650&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: single reading column (~72ch) with full-width data blocks; findings first, evidence after. */
:root {{
  --bg: #f5f6f2; --surface: #ffffff; --fg: #1d211c; --muted: #5a6157; --rule: #dcdfd6;
  --accent: #3d6b2f; --bar: #2a78d6; --track: #e8ebe3; --neg: #c2502e;
  --f-display: "Literata", Georgia, serif; --f-body: "Public Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #151814; --surface: #1d211c; --fg: #e7eae2; --muted: #a3ab9d; --rule: #343a31;
  --accent: #8fc277; --bar: #3987e5; --track: #2a3027; --neg: #e2795a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #151814; --surface: #1d211c; --fg: #e7eae2; --muted: #a3ab9d; --rule: #343a31;
  --accent: #8fc277; --bar: #3987e5; --track: #2a3027; --neg: #e2795a; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--f-body); padding-inline: 20px; padding-block: 40px 64px; }}
main {{ max-width: 980px; margin: 0 auto; display: grid; gap: 44px; }}
.col {{ max-width: 72ch; }}
h1, h2, h3 {{ font-family: var(--f-display); text-wrap: balance; line-height: 1.2; margin: 0; }}
h1 {{ font-size: clamp(28px, 4vw, 40px); font-weight: 650; }}
h2 {{ font-size: 23px; font-weight: 650; padding-bottom: 8px; border-bottom: 1px solid var(--rule); }}
h3 {{ font-size: 17px; font-weight: 500; }}
p {{ margin: 0; }}
section {{ display: grid; gap: 16px; min-width: 0; }}
.eyebrow {{ font: 500 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
.lede {{ color: var(--muted); font-size: 16px; max-width: 72ch; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 10px 28px; font: 13px var(--f-mono); color: var(--muted); }}
.facts b {{ color: var(--fg); font-weight: 500; }}
ol.find {{ margin: 0; padding-left: 1.3em; display: grid; gap: 12px; max-width: 76ch; }}
ol.find li::marker {{ font-family: var(--f-mono); color: var(--accent); }}
ol.find b {{ font-weight: 600; }}
.tw {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ padding: 7px 12px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
th {{ font: 600 12px/1.3 var(--f-body); color: var(--muted); background: var(--bg); vertical-align: bottom; white-space: normal; }}
th:first-child, td.t {{ text-align: left; }}
td.n {{ font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: 12.5px; }}
tbody tr:last-child td {{ border-bottom: 0; }}
table.wide td {{ font-size: 12px; padding: 6px 10px; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; align-items: start; }}
.two > * {{ min-width: 0; }}
.fig {{ display: grid; gap: 8px; }}
.fig figcaption, .note {{ font-size: 13px; color: var(--muted); }}
figure {{ margin: 0; }}
svg.bars {{ width: 100%; max-width: 520px; height: auto; display: block; }}
svg .bl {{ font: 12px var(--f-body); fill: var(--fg); }}
svg .bv {{ font: 12px var(--f-mono); fill: var(--muted); }}
svg .bar {{ fill: var(--bar); }} svg .track {{ fill: var(--track); }}
.swatches {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }}
.sw {{ display: flex; gap: 10px; align-items: flex-start; }}
.sw > div {{ display: grid; gap: 2px; min-width: 0; }}
.chip {{ width: 44px; height: 44px; border-radius: 4px; flex: none; border: 1px solid var(--rule); }}
.m {{ font: 11.5px var(--f-mono); color: var(--muted); }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; margin: 0; font-size: 14px; max-width: 80ch; }}
dt {{ font-weight: 600; }} dd {{ margin: 0; color: var(--muted); }}
.rho {{ display: inline-block; width: 90px; height: 8px; background: var(--track); border-radius: 2px; position: relative; vertical-align: middle; }}
.rho span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px; }}
.rho .pos {{ background: var(--bar); }} .rho .neg {{ background: var(--neg); }}
.callout {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 14px; max-width: 76ch; color: var(--fg); }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; }} dd {{ margin-bottom: 6px; }} }}
</style>
<main>
<header style="display:grid;gap:14px">
  <span class="eyebrow">Delineate Anything · {esc(args.model)} · Ghana 1 m test set</span>
  <h1>{esc(args.title)}</h1>
  <p class="lede">What the reference fields and the predicted fields look like, and which of their characteristics go with
  good or poor delineation. Model: Delineate Anything v2 {esc(args.model_desc)} (<code>{esc(args.model)}</code>),
  run on the 32 Ghana test images.</p>
  <div class="facts"><span><b>32</b> images</span><span><b>{num(n_f)}</b> reference fields</span><span><b>{num(n_p)}</b> predicted polygons</span>
  <span>regions <b>north (21)</b>, <b>ej (11)</b></span><span>field F1 <b>{I.tp.sum() / ((I.n_pred.sum() + I.n_fields.sum()) / 2):.3f}</b></span></div>
</header>

<section>
  <h2>Key findings</h2>
  <ol class="find">
    <li><b>Field size matters most.</b> Only {pct(small_found)} of the smallest quarter of fields (under about 740 m², roughly 27 × 27 m) are found,
    against {pct(big_found)} of the largest quarter. Small fields are mostly absorbed into bigger predictions ({pct(small_merged)} merged); {big_fate}.</li>
    <li><b>Most fields are seen, but not outlined as one piece.</b> On average {pct(covered)} of each reference field's area lies under some prediction, yet only
    {pct(found)} are matched one-to-one at IoU ≥ 0.5. The gap is splitting ({pct(split)} of fields) and merging ({pct(merged)}), not missing fields.</li>
    {crop_finding}
    <li><b>Label quality shows in the scores.</b> Fields labelled with high confidence are found {pct(conf.found['high'])} of the time; low-confidence or unlabelled-confidence fields only
    {pct(conf.found['low'])}–{pct(conf.found['none'])}. "Unidentified" and "fallow" fields are found less often than maize and are merged more often ({pct(crop.merged['unidentified'])} of unidentified fields).</li>
    <li><b>{pct(nc_share)} of the predicted polygons fall on the non-crop background</b> ({num(kind_n['mostly in non-crop area'])} of {num(n_p)}). Another
    {pct(kinds.set_index('kind').share['piece of a field / poor outline'])} lie inside real fields but don't match them: pieces of split fields. Only
    {pct(kinds.set_index('kind').share['matched field'])} of predictions match a field.</li>
    <li><b>Images with dense, regular field patterns score best.</b> Across images, F1 rises with fields per hectare (ρ = {corr.set_index('factor').spearman_rho['fields_per_ha']:+.2f})
    and falls with the share of non-crop land (ρ = {corr.set_index('factor').spearman_rho['noncrop_share']:+.2f}). Colour and brightness of the image matter little.</li>
    {CONF_FINDING}
  </ol>
</section>

{MAP_SECTION}
<section>
  <h2>What information exists</h2>
  <p class="col">{CONF_INTRO}</p>
  <dl>
    <dt>Labels (GPS survey)</dt><dd>Crop (<code>crop</code>: maize, rice, other, noncrop) and detailed crop (<code>crop1</code>: 24 values incl. fallow and unidentified),
    label confidence (high / medium / low / none), region (north, ej), survey date (15 Sep 2021 in the north, 15 Dec 2021 in ej, a few others).</dd>
    <dt>Images (1 m RGB)</dt><dd>Measured per field: mean colour, brightness, greenness (VARI, from −1 to 1; higher is greener), texture (spread of brightness inside the field)
    and edge contrast (brightness difference between the field and a 3 m ring around it). Image acquisition dates are not available; colours describe the imagery, not the survey day.</dd>
    <dt>Scores</dt><dd>Per image: precision, recall, F1, area omission/commission, boundary F1, PoLiS. Per field: matched at IoU ≥ 0.5, best IoU, area covered, split, merged.</dd>
    {CONF_DT}
  </dl>
  <p class="note">How to read the outcome columns: <b>Found</b> = matched one-to-one with IoU ≥ 0.5 · <b>Area covered</b> = share of the field under any prediction ·
  <b>Split</b> = covered by two or more predictions · <b>Merged</b> = its main prediction also covers another field.</p>
</section>

<section>
  <h2>Field size</h2>
  <div class="two">
    <figure class="fig">{bars(size_rows, "v", "l", maxv=1)}
      <figcaption>Share of reference fields found, by size quarter (each quarter ≈ 384 fields).</figcaption></figure>
    <p class="col">{size_text}</p>
  </div>
  {table(size_df, [size_col] + outcome_cols, ["Field size"] + outcome_heads, [str] + outcome_fmts)}
</section>

<section>
  <h2>Crop type</h2>
  <div class="two">
    <figure class="fig">{bars(crop_rows, "v", "l", maxv=1)}
      <figcaption>Share of reference fields found, by crop group.</figcaption></figure>
    <div class="swatches">{"".join(sw)}</div>
  </div>
  <p class="note">Swatches show each group's median field colour in the imagery. The model only learns "field", never the crop; differences come from how each crop looks and how big its fields are.
  Rice: almost all in the north. Other crops: {crop.n['other crops']:.0f} fields, mostly sorghum, yam, small mixed plots, groundnuts and cassava.</p>
  {outcome_table("by_crop_group", "Crop group", crop_names)}
</section>

<section>
  <h2>Label confidence and region</h2>
  <div class="two">
    {outcome_table("by_label_confidence", "Label confidence", {"none": "not given"})}
    {outcome_table("by_region", "Region", {"ej": "ej (surveyed Dec 2021)", "north": "north (surveyed Sep 2021)"})}
  </div>
  <p class="col">Fields the surveyors were sure about are found about {conf_ratio:.1f} times as often as the uncertain ones (low or no confidence). Part of that is the model, part is probably the label itself:
  an uncertain or unidentified polygon is more likely to be something other than a clear field. The two regions perform alike; region and survey date coincide, so their effects can't be separated.</p>
</section>

<section>
  <h2>How fields look</h2>
  <p class="col">Each table splits the {num(n_f)} reference fields into four equal groups by one measured property.</p>
  <div class="two">
    <div class="fig"><h3>Greenness (VARI)</h3>{table(vari_df, [vari_col, "n", "found", "covered", "split", "merged"], ["VARI", "Fields", "Found", "Covered", "Split", "Merged"], [str, num, pct, pct, pct, pct])}</div>
    <div class="fig"><h3>Texture inside the field</h3>{table(tex_df, [tex_col, "n", "found", "covered", "split", "merged"], ["Texture", "Fields", "Found", "Covered", "Split", "Merged"], [str, num, pct, pct, pct, pct])}</div>
    <div class="fig"><h3>Edge contrast with surroundings</h3>{table(edge_df, [edge_col, "n", "found", "covered", "split", "merged"], ["Contrast", "Fields", "Found", "Covered", "Split", "Merged"], [str, num, pct, pct, pct, pct])}</div>
    <div class="fig"><h3>Shape (compactness)</h3>{table(comp_df, [comp_col, "n", "found", "covered", "split", "merged"], ["Compactness", "Fields", "Found", "Covered", "Split", "Merged"], [str, num, pct, pct, pct, pct])}</div>
  </div>
  <p class="col">{vari_text} {tex_text} {edge_text} {comp_text} {bri_text}</p>
</section>

<section>
  <h2>What the predicted polygons are</h2>
  <div class="two">
    <figure class="fig">{bars(kind_rows, "v", "k", maxv=0.5)}
      <figcaption>Share of the {num(n_p)} predicted polygons in each category.</figcaption></figure>
    <p class="col">Only {pct(kinds.set_index('kind').share['matched field'])} of predictions are a clean match. {piece_word} are pieces of real fields: the model sees those fields, but draws
    them in parts. {pct(nc_share)} sit on the non-crop background; these are the real false detections according to the labels. They are darker than the other predictions
    and have weaker edges than the matched ones. The predictions that merge several fields are the largest (median {kinds.set_index('kind').median_area['covers several fields']:,.0f} m²).</p>
  </div>
</section>

{CONF_SECTION}
<section>
  <h2>Image by image</h2>
  <div class="two">
    <div class="fig"><h3>What goes with a higher F1 across images</h3>
      <div class="tw"><table><thead><tr><th>Image property</th><th>Spearman ρ with F1</th><th></th></tr></thead><tbody>{corr_rows}</tbody></table></div>
      <p class="note">32 images, so only |ρ| above about 0.35 is clearly more than noise. Blue: higher value, higher F1. Red: higher value, lower F1.</p></div>
    <p class="col">{image_text}</p>
  </div>
  {img_table}
  <div class="xy" id="xy-img" data-title="images"></div>
  <p class="note">Sorted by F1. Pick any two columns to plot them against each other; hover a point to see the image. Non-crop share = share of the image inside the non-crop background polygon. Area omission = share of field area not covered by any prediction.</p>
</section>

<section>
  <h2>What this suggests</h2>
  <ol class="find">
    <li><b>Tune the delineation settings before retraining.</b> Splitting of large fields and merging of small ones are the two biggest losses, and both are driven by the
    merge thresholds and minimum areas in the config. Tune them on the validation images.</li>
    <li><b>Decide what counts as a field.</b> Unidentified and low-confidence polygons behave differently from the clearly labelled crops. Scoring with and without them
    shows how much of the gap is label definition.</li>
    <li><b>Look at the greenest and the smallest fields in QGIS.</b> They are where the model fails most; a visual check will tell whether the boundaries are visible at 1 m at all.</li>
    {CONF_SUGGEST}
  </ol>
  <p class="note">Data: <code>{RES}/analysis/</code>: fields.csv (one row per reference field), predictions.csv (one row per predicted polygon),
  images.csv, summary.json. Script: <code>finetune/analyze_predictions.py</code>.</p>
</section>
</main>
"""

# ---------- every polygon: compact rows embedded as JSON, rendered by a small table script
short = lambda s: s.replace("GHANA_", "").replace("_1m", "")
hexc = lambda r, g, b: "#%02x%02x%02x" % tuple(int(round(min(max(v, 0), 255))) for v in (r, g, b)) if r == r else ""


def field_outcome(r):
    if r.matched: return "Found"
    if r.covered_share < 0.2: return "Missed"
    if r.n_pred_covering >= 2: return "Split"
    if r.merged: return "Merged"
    return "Poor outline"


kind_short = {"matched field": "Matched field", "piece of a field / poor outline": "Piece of a field",
              "mostly in non-crop area": "On non-crop background", "covers several fields": "Covers several fields",
              "mixed / boundary": "Mixed / boundary"}
rnd = lambda v, d: None if v != v else round(float(v), d)
fields_rows = [[short(r.image), str(r.label_id), crop_names[r.crop_group], r.crop1, r.label_confidence, rnd(r.area_m2, 0),
                field_outcome(r), rnd(r.best_iou, 2), rnd(r.covered_share, 2), hexc(r.r, r.g, r.b), rnd(r.vari, 2),
                rnd(r.texture, 1), rnd(r.edge_contrast, 1), rnd(r.compactness, 2)] for r in F.itertuples()]
pred_rows = [[short(r.image), int(r.id), kind_short[r.kind], rnd(r.area_m2, 0), rnd(r.share_in_fields, 2),
              rnd(r.share_in_noncrop, 2), int(r.n_fields_covered), hexc(r.r, r.g, r.b), rnd(r.vari, 2),
              rnd(r.texture, 1), rnd(r.edge_contrast, 1), rnd(r.compactness, 2)] + ([rnd(r.confidence, 3)] if HAS_CONF else []) for r in P.itertuples()]
poly_json = json.dumps({"fields": fields_rows, "preds": pred_rows}, separators=(",", ":")).replace("</", "<\\/")

POLY = """
<section id="polygons">
  <h2>Every polygon</h2>
  <p class="col">One row per reference field or predicted polygon. IDs are the <code>id</code> attribute in the label file
  (<code>labels/filtered_grids(1).gpkg</code>) and in each image's prediction file (<code>__RES_SHORT__/&lt;image&gt;.gpkg</code>),
  so a row can be found in QGIS. Click a column header to sort.</p>
  <div class="tabs" role="tablist">
    <button role="tab" id="tab-fields" aria-selected="true">Reference fields <span class="cnt" id="n-fields"></span></button>
    <button role="tab" id="tab-preds" aria-selected="false">Predicted polygons <span class="cnt" id="n-preds"></span></button>
  </div>
  <div class="filters">
    <label for="f-image">Image <select id="f-image"></select></label>
    <label for="f-cat"><span id="f-cat-label">Outcome</span> <select id="f-cat"></select></label>
    <label for="f-crop" id="f-crop-wrap">Crop <select id="f-crop"></select></label>
    <span class="shown" id="shown"></span>
  </div>
  <div class="tw"><table class="poly" id="poly"><thead></thead><tbody></tbody></table></div>
  <div class="pager"><button id="prev">Previous</button><span id="pageinfo"></span><button id="next">Next</button></div>
  <p class="note" id="legend"></p>
  <h3>Plot two columns</h3>
  <p class="note">Plots the rows the table above shows (same tab and filters). Text columns become bands with points spread across them.</p>
  <div class="xy" id="xy-poly" data-title="polygons"></div>
</section>
"""

SCRIPT = """
<script>
window.makeXY = function (root, onLabel) {
  const pre = root.id;
  root.innerHTML = `<div class="xy-ctl">
    <label for="${pre}-x">X <select id="${pre}-x"></select></label>
    <label for="${pre}-y">Y <select id="${pre}-y"></select></label>
    <label for="${pre}-c">Colour by <select id="${pre}-c"></select></label>
    <label class="chk" for="${pre}-lx"><input type="checkbox" id="${pre}-lx"> log X</label>
    <label class="chk" for="${pre}-ly"><input type="checkbox" id="${pre}-ly"> log Y</label>
    <span class="xy-n" id="${pre}-n"></span></div>
    <div class="xy-plot"><svg class="xy-svg" id="${pre}-svg" viewBox="0 0 680 430" role="img" aria-label="Scatter plot"></svg>
    <div class="xy-tip" id="${pre}-tip" hidden></div></div>
    <div class="legend" id="${pre}-leg"></div>`;
  const $ = s => document.getElementById(pre + "-" + s);
  const W = 680, H = 430, ML = 64, MR = 18, MT = 14, MB = 52;
  let cols = [], rows = [], key = null;
  const ok = c => c[1] !== "c";
  const fmt = (v, k) => v === null || v === undefined || v === "" ? "–" : k === "p" ? Math.round(v * 100) + "%" :
    typeof v === "number" ? v.toLocaleString(undefined, {maximumFractionDigits: Math.abs(v) < 10 ? 3 : 0}) : v;
  function nice(lo, hi, n) {
    const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(st => st >= raw);
    const out = []; for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(+t.toPrecision(12)); return out;
  }
  function axis(i, len, logOn) {
    const [name, kind] = cols[i];
    const vals = rows.map(r => r[i]).filter(v => v !== null && v !== undefined && v !== "");
    if (kind === "t") {
      const cats = [...new Set(vals)].sort();
      const pos = v => (cats.indexOf(v) + 0.5) / cats.length;
      return {cat: true, cats, f: (v, j) => (pos(v) + ((((j * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * 0.6 / cats.length) * len,
              ticks: cats.map(c => [c, pos(c) * len]), name};
    }
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const canLog = lo > 0;
    if (logOn && canLog) {
      const a = Math.log10(lo), b = Math.log10(hi), pad = (b - a || 1) * 0.04;
      const f = v => (Math.log10(v) - (a - pad)) / ((b + pad) - (a - pad)) * len;
      const ticks = []; for (let e = Math.floor(a); e <= Math.ceil(b); e++) for (const m of [1, 2, 5]) { const t = m * Math.pow(10, e); if (t >= lo / 1.1 && t <= hi * 1.1) ticks.push([fmt(t, kind), f(t)]); }
      return {f: (v) => v > 0 ? f(v) : null, ticks, name, canLog, kind};
    }
    const pad = (hi - lo || 1) * 0.04; lo -= pad; hi += pad;
    const f = v => (v - lo) / (hi - lo) * len;
    return {f, ticks: nice(lo, hi, 5).map(t => [fmt(t, kind), f(t)]), name, canLog, kind};
  }
  function spearman(a, b) {
    const rk = x => { const o = x.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = new Array(x.length);
      for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; } return r; };
    const ra = rk(a), rb = rk(b), n = a.length, ma = (n - 1) / 2;
    let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - ma); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - ma) ** 2; }
    return sab / Math.sqrt(saa * sbb);
  }
  function draw() {
    const xi = +$("x").value, yi = +$("y").value, ci = $("c").value === "" ? -1 : +$("c").value;
    const pw = W - ML - MR, ph = H - MT - MB;
    const X = axis(xi, pw, $("lx").checked), Y = axis(yi, ph, $("ly").checked);
    $("lx").disabled = !!X.cat || !X.canLog; $("ly").disabled = !!Y.cat || !Y.canLog;
    const cats = ci >= 0 ? [...new Set(rows.map(r => r[ci]))].sort() : [];
    const o = [];
    for (const [lab, p] of X.ticks) {
      o.push(`<line x1="${ML + p}" x2="${ML + p}" y1="${MT}" y2="${MT + ph}" class="g"/>`);
      o.push(`<text x="${ML + p}" y="${MT + ph + 16}" text-anchor="middle" class="ax">${String(lab).length > 12 ? String(lab).slice(0, 11) + "…" : lab}</text>`);
    }
    for (const [lab, p] of Y.ticks) {
      o.push(`<line x1="${ML}" x2="${ML + pw}" y1="${MT + ph - p}" y2="${MT + ph - p}" class="g"/>`);
      o.push(`<text x="${ML - 6}" y="${MT + ph - p + 4}" text-anchor="end" class="ax">${String(lab).length > 9 ? String(lab).slice(0, 8) + "…" : lab}</text>`);
    }
    o.push(`<text x="${ML + pw / 2}" y="${H - 8}" text-anchor="middle" class="axt">${X.name}</text>`);
    o.push(`<text transform="translate(14 ${MT + ph / 2}) rotate(-90)" text-anchor="middle" class="axt">${Y.name}</text>`);
    let n = 0, skipped = 0; const xs = [], ys = [];
    rows.forEach((r, j) => {
      const xv = r[xi], yv = r[yi];
      if (xv === null || yv === null || xv === "" || yv === "") { skipped++; return; }
      const px = X.f(xv, j), py = Y.f(yv, j + 7919);
      if (px === null || py === null) { skipped++; return; }
      const k = ci >= 0 ? cats.indexOf(r[ci]) : 0;
      o.push(`<circle cx="${(ML + px).toFixed(1)}" cy="${(MT + ph - py).toFixed(1)}" r="${rows.length > 200 ? 3.4 : 5.5}" class="pt c${k < 8 ? k + 1 : "x"}" data-j="${j}"/>`);
      n++; if (!X.cat && !Y.cat) { xs.push(xv); ys.push(yv); }
    });
    $("svg").innerHTML = o.join("");
    const rho = xs.length > 3 ? spearman(xs, ys) : null;
    $("n").textContent = `${n.toLocaleString()} points` + (skipped ? ` · ${skipped} without values` : "") + (rho !== null && isFinite(rho) ? ` · Spearman ρ = ${rho.toFixed(2)}` : "");
    $("leg").innerHTML = ci >= 0 ? cats.map((c, k) => `<span><i class="c${k < 8 ? k + 1 : "x"}"></i>${c}</span>`).join("") + (cats.length > 8 ? `<span>(colours repeat grey after 8 values)</span>` : "") : "";
    $("svg").onmousemove = e => {
      const t = e.target.closest("circle"); const tip = $("tip");
      if (!t) { tip.hidden = true; return; }
      const r = rows[+t.dataset.j];
      tip.innerHTML = `<b>${onLabel(r)}</b><br>${cols[xi][0]}: ${fmt(r[xi], cols[xi][1])}<br>${cols[yi][0]}: ${fmt(r[yi], cols[yi][1])}` + (ci >= 0 ? `<br>${cols[ci][0]}: ${r[ci]}` : "");
      const box = root.querySelector(".xy-plot").getBoundingClientRect();
      tip.hidden = false;
      const left = e.clientX - box.left + 12, top = e.clientY - box.top + 12;
      tip.style.left = Math.min(left, box.width - tip.offsetWidth - 4) + "px"; tip.style.top = Math.min(top, box.height - tip.offsetHeight - 4) + "px";
    };
    $("svg").onmouseleave = () => { $("tip").hidden = true; };
  }
  ["x", "y", "c", "lx", "ly"].forEach(id => $(id).addEventListener("change", draw));
  return {
    update(newCols, newRows, newKey, defaults) {
      rows = newRows;
      if (newKey !== key) {
        cols = newCols; key = newKey;
        const axisOpts = cols.map((c, i) => ok(c) ? `<option value="${i}">${c[0]}</option>` : "").join("");
        $("x").innerHTML = axisOpts; $("y").innerHTML = axisOpts;
        const catCols = cols.map((c, i) => [c, i]).filter(([c, i]) => c[1] === "t" && new Set(rows.map(r => r[i])).size <= 12);
        $("c").innerHTML = `<option value="">None</option>` + catCols.map(([c, i]) => `<option value="${i}">${c[0]}</option>`).join("");
        const idx = name => cols.findIndex(c => c[0] === name);
        $("x").value = idx(defaults.x); $("y").value = idx(defaults.y); $("c").value = defaults.c ? idx(defaults.c) : "";
        $("lx").checked = !!defaults.lx; $("ly").checked = !!defaults.ly;
      }
      draw();
    }
  };
};
</script>

<script type="application/json" id="imgdata">__IMGDATA__</script>
<script type="application/json" id="polydata">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('polydata').textContent);
  const COLS = {
    fields: [["Image","t"],["Label ID","t"],["Crop","t"],["Crop (detail)","t"],["Label conf.","t"],["Area (m²)","n"],["Outcome","t"],
             ["Best IoU","n"],["Covered","p"],["Colour","c"],["VARI","n"],["Texture","n"],["Edge contrast","n"],["Compactness","n"]],
    preds: [["Image","t"],["Polygon ID","n"],["Category","t"],["Area (m²)","n"],["In fields","p"],["In non-crop","p"],["Fields covered","n"],
            ["Colour","c"],["VARI","n"],["Texture","n"],["Edge contrast","n"],["Compactness","n"]__CONFCOL__]};
  const CAT = {fields: 6, preds: 2};
  const LEGEND = {
    fields: "Outcome: Found = matched one-to-one at IoU ≥ 0.5 · Missed = less than 20% of the field under any prediction · Split = covered by two or more predictions · Merged = its main prediction also covers another field · Poor outline = covered, but no prediction fits it well. Covered = share of the field's area under any prediction. Colour = mean colour of the field in the image.",
    preds: "Category: Matched field = matches one reference field at IoU ≥ 0.5 · Piece of a field = mostly inside reference fields but no match · On non-crop background = at least half on the non-crop polygon · Covers several fields = covers two or more reference fields · Mixed = straddles fields and background. In fields / In non-crop = share of the polygon's area."};
  const PAGE = 50;
  const polyXY = makeXY(document.getElementById("xy-poly"), r => `${r[0]} · ID ${r[1]}`);
  const IMG = JSON.parse(document.getElementById("imgdata").textContent);
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
                                                         : {x: "Area (m²)", y: cols.some(c => c[0] === "Confidence") ? "Confidence" : "In non-crop", c: "Category", lx: true});
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
})();
</script>
"""

CSS = """
:root { --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7; --s8: #e34948; --sx: #9a9a94; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767; --sx: #77776f; } }
:root[data-theme="dark"] { --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767; --sx: #77776f; }
.xy { display: grid; gap: 10px; }
.xy-ctl { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; font-size: 13px; color: var(--muted); }
.xy-ctl select { font: 13px var(--f-body); color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 4px 6px; margin-left: 4px; max-width: 60vw; }
.xy-ctl .chk { display: inline-flex; gap: 4px; align-items: center; }
.xy-n { font-family: var(--f-mono); font-size: 12px; margin-left: auto; }
.xy-plot { position: relative; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); padding: 6px; }
.xy-svg { width: 100%; height: auto; display: block; }
.xy-svg .g { stroke: var(--rule); stroke-width: 1; }
.xy-svg .ax { font: 10.5px var(--f-mono); fill: var(--muted); }
.xy-svg .axt { font: 600 12px var(--f-body); fill: var(--fg); }
.xy-svg .pt { stroke: var(--surface); stroke-width: 1; fill-opacity: .85; cursor: crosshair; }
.xy-svg .pt:hover { fill-opacity: 1; stroke: var(--fg); }
.xy-tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font: 12px/1.45 var(--f-body); padding: 6px 9px; border-radius: 4px; max-width: 260px; z-index: 2; }
.c1 { fill: var(--s1); background: var(--s1); } .c2 { fill: var(--s2); background: var(--s2); } .c3 { fill: var(--s3); background: var(--s3); }
.c4 { fill: var(--s4); background: var(--s4); } .c5 { fill: var(--s5); background: var(--s5); } .c6 { fill: var(--s6); background: var(--s6); }
.c7 { fill: var(--s7); background: var(--s7); } .c8 { fill: var(--s8); background: var(--s8); } .cx { fill: var(--sx); background: var(--sx); }
svg.sweep { width: 100%; max-width: 560px; height: auto; display: block; }
svg.sweep .grid { stroke: var(--rule); stroke-width: 1; }
svg.sweep .cur { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
svg.sweep .ax { font: 11px var(--f-mono); fill: var(--muted); }
svg.sweep .ln { fill: none; stroke-width: 2; }
svg.sweep .pt { stroke: var(--surface); stroke-width: 2; }
svg.sweep .lab { font: 600 12px var(--f-body); fill: var(--fg); }
svg.sweep .s1 { stroke: var(--s1); } svg.sweep circle.s1 { fill: var(--s1); }
svg.sweep .s2 { stroke: var(--s2); } svg.sweep circle.s2 { fill: var(--s2); }
svg.sweep .s3 { stroke: var(--s3); } svg.sweep circle.s3 { fill: var(--s3); }
svg.sweep circle { stroke: var(--surface); }
tr.cur td { background: var(--track); font-weight: 600; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12.5px; color: var(--fg); }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--rule); }
.tabs button { font: 600 14px var(--f-body); color: var(--muted); background: none; border: 0; border-bottom: 2px solid transparent; padding: 8px 12px; cursor: pointer; }
.tabs button[aria-selected="true"] { color: var(--fg); border-bottom-color: var(--accent); }
.tabs .cnt { font: 12px var(--f-mono); color: var(--muted); margin-left: 4px; }
.filters { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; font-size: 13px; color: var(--muted); }
.filters select { font: 13px var(--f-body); color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 4px 6px; margin-left: 4px; max-width: 70vw; }
.shown { font-family: var(--f-mono); margin-left: auto; }
table.poly th { padding: 0; }
table.poly th button { all: unset; display: block; padding: 7px 10px; cursor: pointer; font: 600 12px/1.3 var(--f-body); color: var(--muted); white-space: nowrap; }
table.poly th.num button { text-align: right; }
table.poly td { font-size: 12px; padding: 5px 10px; }
.dot { display: inline-block; width: 16px; height: 16px; border-radius: 3px; border: 1px solid var(--rule); vertical-align: middle; }
.pager { display: flex; gap: 12px; align-items: center; font: 13px var(--f-mono); color: var(--muted); }
.pager button { font: 13px var(--f-body); color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 5px 12px; cursor: pointer; }
.pager button:disabled { opacity: .45; cursor: default; }
button:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""

POLY = POLY.replace("__RES_SHORT__", RES_SHORT)
page = page.replace("</style>", CSS + "</style>", 1)
page = page.replace("</main>\n", POLY + "</main>\n" + SCRIPT.replace("__IMGDATA__", img_json).replace("__DATA__", poly_json).replace("__CONFCOL__", ',["Confidence","n"]' if HAS_CONF else ""), 1)
page = page.replace("images.csv, summary.json. Script:", "images.csv, summary.json. The table of every polygon is at the end of this page. Script:")
open(OUT, "w").write(page)
print("written", len(page))

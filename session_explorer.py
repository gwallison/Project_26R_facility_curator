"""
session_explorer.py — Streamlit explorer for facility curation session results.

Reads every sessions/<slug>/output/ directory (lab_results.parquet +
approved_links.csv), stacks them into one frame tagged by facility, and lets you
slice by facility, analyte, result flag, matrix, and well pad.

Run: streamlit run session_explorer.py
"""

import json
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Facility Session Explorer",
    page_icon="🏭",
    layout="wide",
)

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"

GCS_BASE = "https://storage.googleapis.com/fta-form26r-library"
LOCAL_PDF_ROOT = Path(r"D:\PA_Form26r_PDFs\all_pdfs")
LOCAL_PDF_PORT = 8765

NUMERIC_FLAGS = {"=", "<", ">", "~"}

FLAG_LABELS = {
    "=": "= Detected",
    "<": "< Non-detect",
    ">": "> Above limit",
    "~": "~ Estimated (J)",
    "Q": "Q Qualitative",
    "T": "T Text ref",
    "?": "? Unparseable",
}

# Categorical slots, assigned in fixed order and never cycled — a 9th group
# folds into "Other".  Both modes are selected, not flipped: each is stepped for
# its own surface and validated there (light — worst adjacent CVD ΔE 9.1,
# normal-vision ΔE 19.6; dark — CVD ΔE 8.4, normal-vision ΔE 19.3).
LIGHT = dict(
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    neutral="#8a8a85", surface="#fcfcfb", grid="#ececeb",
    ink="#0b0b0b", muted="#52514e",
)
DARK = dict(
    series=["#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"],
    neutral="#9a998f", surface="#1a1a19", grid="#33322f",
    ink="#ffffff", muted="#c3c2b7",
)

OTHER_LABEL = "Other"
MAX_SERIES = len(LIGHT["series"])


def palette() -> dict:
    """Palette for the viewer's active Streamlit theme."""
    mode = "light"
    try:
        mode = st.context.theme.type or "light"
    except Exception:
        pass
    return DARK if mode == "dark" else LIGHT


def flag_colors(p: dict) -> dict:
    """Colour follows the entity, never its rank: flags keep fixed hues."""
    s, n = p["series"], p["neutral"]
    return {
        "=": s[0],   # detected
        "<": n,      # non-detect — genuinely the "no signal" category
        ">": s[1],   # above limit
        "~": s[6],   # estimated
        "Q": s[2],
        "T": s[5],
        "?": n,
    }

_MATRIX_MAP = {
    "solid": "Solid",
    "solid/grab": "Solid",
    "solid/composite": "Solid",
    "soil": "Solid",
    "sludge": "Solid",
    "sediment": "Solid",
    "water": "Water",
    "water/grab": "Water",
    "water/composite": "Water",
    "aqueous": "Water",
    "non-potable water": "Water",
    "non potable water": "Water",
    "grab": "Water",
    "waste water": "Wastewater",
    "wastewater": "Wastewater",
    "waste - liquid": "Wastewater",
    "liquid": "Wastewater",
    "leachate": "Leachate",
    "extract": "Extract",
    "waste": "Waste (other)",
    "other": "Other",
}

COLOR_DIMS = {
    "result_flag": "Result flag",
    "facility": "Facility",
    "matrix_norm": "Matrix",
    "pad_label": "Well pad",
    "f26r_waste_code": "Waste code",
}


def normalize_matrix(s: pd.Series) -> pd.Series:
    lowered = s.fillna("unknown").str.strip().str.lower()
    return lowered.map(_MATRIX_MAP).fillna(s.fillna("Unknown").str.strip())


# ---------------------------------------------------------------------------
# Local PDF server (shares the root and port used by facility_curator.py)
# ---------------------------------------------------------------------------
def _start_pdf_server():
    if getattr(_start_pdf_server, "_started", False):
        return
    if not LOCAL_PDF_ROOT.is_dir():
        return
    handler = partial(SimpleHTTPRequestHandler, directory=str(LOCAL_PDF_ROOT))
    try:
        server = HTTPServer(("127.0.0.1", LOCAL_PDF_PORT), handler)
    except OSError:
        # Port already bound — facility_curator.py is very likely serving the
        # same root, so local links still resolve.
        _start_pdf_server._started = True
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _start_pdf_server._started = True


_start_pdf_server()


def pdf_url(set_name, filename, page) -> str:
    sn = quote(str(set_name or ""), safe="")
    fn = quote(str(filename or ""), safe="")
    try:
        pg = int(page) if pd.notna(page) else 1
    except (ValueError, TypeError):
        pg = 1
    if getattr(_start_pdf_server, "_started", False):
        return f"http://127.0.0.1:{LOCAL_PDF_PORT}/{sn}/{fn}#page={pg}"
    return f"{GCS_BASE}/full-set/{sn}/{fn}#page={pg}"


def pdf_url_col(df: pd.DataFrame, page_col: str) -> pd.Series:
    return pd.Series(
        [pdf_url(s, f, p) for s, f, p in
         zip(df["set_name"], df["original_filename"], df[page_col])],
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
def session_signature() -> tuple:
    """Cache key: every session file that feeds the app, plus mtime/size."""
    sig = []
    for pat in ("*/output/*", "*/session.json", "*/candidate_pads.csv"):
        for p in sorted(SESSIONS_DIR.glob(pat)):
            stat = p.stat()
            sig.append((str(p.relative_to(SESSIONS_DIR)),
                        stat.st_mtime_ns, stat.st_size))
    return tuple(sig)


@st.cache_data(show_spinner=False)
def load_sessions(sig: tuple):
    """Return (results, links, pads, meta) stacked across all sessions."""
    results, links, pads, meta = [], [], [], []

    for sdir in sorted(SESSIONS_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        slug = sdir.name
        info = {}
        sjson = sdir / "session.json"
        if sjson.exists():
            try:
                info = json.loads(sjson.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                info = {}
        facility = (info.get("facility_name") or slug).strip()

        pad_names = {}
        pad_csv = sdir / "candidate_pads.csv"
        if pad_csv.exists():
            pads_df = pd.read_csv(pad_csv, dtype={"pad_WELL_PAD_ID": str})
            pads_df["session"] = slug
            pads_df["facility"] = facility
            pads.append(pads_df)
            pad_names = dict(zip(
                pads_df["pad_WELL_PAD_ID"].astype(str),
                pads_df["pad_WELL_PAD"].fillna("").astype(str),
            ))

        lab = sdir / "output" / "lab_results.parquet"
        if lab.exists():
            df = pd.read_parquet(lab)
            df["session"] = slug
            df["facility"] = facility
            pid = df["pad_WELL_PAD_ID"].astype(str)
            df["pad_label"] = [
                f"{pad_names[p]} ({p})" if pad_names.get(p) else p for p in pid
            ]
            results.append(df)

        alinks = sdir / "output" / "approved_links.csv"
        if alinks.exists():
            ldf = pd.read_csv(alinks, dtype={"pad_WELL_PAD_ID": str,
                                             "lab_report_id": str,
                                             "report_id": str})
            ldf["session"] = slug
            ldf["facility"] = facility
            ldf["pad_name"] = ldf["pad_WELL_PAD_ID"].astype(str).map(
                lambda p: pad_names.get(p, "")
            )
            links.append(ldf)

        outdir = sdir / "output"
        meta.append({
            "session": slug,
            "facility": facility,
            "facility_query": info.get("facility_query", ""),
            "created": info.get("created", ""),
            "last_modified": info.get("last_modified", ""),
            "has_output": lab.exists(),
            "report_pdf": next(
                (p.name for p in sorted(outdir.glob("*.pdf"))), ""
            ) if outdir.is_dir() else "",
        })

    res = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not res.empty:
        res["matrix_norm"] = normalize_matrix(res["matrix"])
        res["f26r_waste_code"] = res["f26r_waste_code"].fillna("(unassigned)")
        res["collection_dt"] = pd.to_datetime(res["collection_date"],
                                              errors="coerce")

    lnk = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
    pdz = pd.concat(pads, ignore_index=True) if pads else pd.DataFrame()
    return res, lnk, pdz, pd.DataFrame(meta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detection_freq(sub: pd.DataFrame) -> str:
    n = len(sub)
    if n == 0:
        return "—"
    n_det = int((sub["result_flag"] == "=").sum())
    return f"{n_det / n * 100:.1f}% ({n_det:,} / {n:,})"


def top_units(sub: pd.DataFrame, n: int = 5) -> str:
    vc = sub["units"].value_counts().head(n)
    return "  ·  ".join(f"{u} ({c:,})" for u, c in vc.items())


def color_groups(series: pd.Series, color_by: str, p: dict):
    """Fixed-order groups + their colours; folds a long tail into 'Other'."""
    if color_by == "result_flag":
        fc = flag_colors(p)
        present = set(series)
        order = [f for f in FLAG_LABELS if f in present]
        order += sorted(present - set(FLAG_LABELS))
        return order, {g: fc.get(g, p["neutral"]) for g in order}, dict(FLAG_LABELS)

    counts = series.value_counts()
    keep = list(counts.index[:MAX_SERIES])
    colors = {g: p["series"][i] for i, g in enumerate(keep)}
    if len(counts) > MAX_SERIES:
        keep.append(OTHER_LABEL)
        colors[OTHER_LABEL] = p["neutral"]
    return keep, colors, {}


def fold_series(series: pd.Series, keep: list) -> pd.Series:
    if OTHER_LABEL not in keep:
        return series
    allowed = set(keep) - {OTHER_LABEL}
    return series.where(series.isin(allowed), OTHER_LABEL)


def _style(fig: go.Figure, p: dict, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white" if p is LIGHT else "plotly_dark",
        font=dict(color=p["ink"]),
        hovermode="closest",
        margin=dict(t=60, b=50, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=p["ink"])),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False,
                     linecolor=p["grid"], tickfont=dict(color=p["muted"]))
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False,
                     linecolor=p["grid"], tickfont=dict(color=p["muted"]))
    return fig


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def build_figure(sub, analyte, log_scale, chart_type, color_by):
    p = palette()
    surface = p["surface"]
    numeric = sub[sub["result_flag"].isin(NUMERIC_FLAGS)].copy()
    if numeric.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric rows match the current filters.",
                           showarrow=False, font=dict(size=16, color=p["muted"]))
        return _style(fig, p, 300)

    groups, colors, label_map = color_groups(numeric[color_by], color_by, p)
    numeric["_grp"] = fold_series(numeric[color_by], groups)

    def lbl(g):
        return label_map.get(g, str(g))

    fig = go.Figure()
    height = 480
    xaxis_title = analyte

    if chart_type == "Histogram":
        # Overlaid bars occlude each other, so draw the biggest group first and
        # the smallest last.  Colour and legend order still follow the entity —
        # only the z-order tracks size.
        sizes = numeric["_grp"].value_counts()
        draw_order = sorted(groups, key=lambda g: -sizes.get(g, 0))
        for grp in draw_order:
            vals = numeric.loc[numeric["_grp"] == grp, "result_value"].dropna()
            if vals.empty:
                continue
            if log_scale:
                vals = np.log10(vals.clip(lower=1e-12))
                xaxis_title = f"log₁₀({analyte})"
            fig.add_trace(go.Histogram(
                x=vals, name=lbl(grp), opacity=0.75, nbinsx=50,
                legendrank=groups.index(grp),
                marker=dict(color=colors[grp],
                            line=dict(color=surface, width=1)),
                hovertemplate=f"{lbl(grp)}<br>%{{x}}<br>%{{y:,}} rows<extra></extra>",
            ))
        fig.update_layout(barmode="overlay", xaxis_title=xaxis_title,
                          yaxis_title="Count")

    elif chart_type == "Box plot":
        for grp in groups:
            vals = numeric.loc[numeric["_grp"] == grp, "result_value"].dropna()
            if vals.empty:
                continue
            fig.add_trace(go.Box(
                y=vals, name=lbl(grp), boxpoints="outliers",
                marker=dict(color=colors[grp], size=8,
                            line=dict(color=surface, width=1)),
                line=dict(width=2),
            ))
        fig.update_layout(yaxis_title=analyte,
                          yaxis_type="log" if log_scale else "linear")

    elif chart_type == "Strip / dot plot":
        # Identity is carried by the y-axis label as well as by hue, so this
        # form stays readable past the all-pairs colour cap.
        for grp in groups:
            rows = numeric[numeric["_grp"] == grp]
            if rows.empty:
                continue
            first = True
            for flag, symbol in [("=", "circle"), ("<", "triangle-down"),
                                 (">", "triangle-up"), ("~", "diamond")]:
                pts = rows[rows["result_flag"] == flag]
                if pts.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=pts["result_value"], y=[lbl(grp)] * len(pts),
                    mode="markers", name=lbl(grp), legendgroup=str(grp),
                    showlegend=first,
                    marker=dict(symbol=symbol, size=8, opacity=0.55,
                                color=colors[grp],
                                line=dict(color=surface, width=2)),
                    customdata=pts[["lab_sample_id", "units", "result"]].values,
                    hovertemplate=(
                        f"{lbl(grp)} · {FLAG_LABELS.get(flag, flag)}<br>"
                        "%{customdata[2]} %{customdata[1]}<br>"
                        "sample %{customdata[0]}<extra></extra>"
                    ),
                ))
                first = False
        height = max(320, len(groups) * 60 + 140)
        fig.update_layout(xaxis_title=analyte,
                          xaxis_type="log" if log_scale else "linear")

    elif chart_type == "CDF":
        for grp in groups:
            vals = numeric.loc[numeric["_grp"] == grp,
                               "result_value"].dropna().sort_values()
            if vals.empty:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            fig.add_trace(go.Scatter(
                x=vals, y=y, mode="lines", name=lbl(grp),
                line=dict(color=colors[grp], width=2),
                hovertemplate=f"{lbl(grp)}<br>%{{x}}<br>%{{y:.0%}}<extra></extra>",
            ))
        fig.update_layout(xaxis_title=analyte,
                          xaxis_type="log" if log_scale else "linear",
                          yaxis_title="Cumulative fraction",
                          yaxis_range=[0, 1])

    fig.update_layout(
        title=f"{analyte}  —  {len(numeric):,} numeric rows",
        legend_title=COLOR_DIMS.get(color_by, color_by),
        showlegend=len(fig.data) > 1,
    )
    return _style(fig, p, height)


def hbar(counts: pd.Series, title: str, xlabel: str) -> go.Figure:
    """Magnitude by category — one series, so no legend; direct value labels."""
    p = palette()
    counts = counts.sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=[str(i) for i in counts.index], orientation="h",
        marker=dict(color=p["series"][0],
                    line=dict(color=p["surface"], width=2)),
        text=[f"{v:,}" for v in counts.values], textposition="outside",
        textfont=dict(color=p["muted"]), cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, showlegend=False,
                      margin=dict(t=60, b=50, l=10, r=20))
    fig = _style(fig, p, max(280, len(counts) * 32 + 140))
    # Headroom so the outside value label on the longest bar isn't clipped.
    fig.update_xaxes(range=[0, float(max(counts.values, default=1)) * 1.18])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=p["ink"]))
    return fig


def coverage_heatmap(piv: pd.DataFrame, metric: str) -> go.Figure:
    """Magnitude across two categorical axes — one hue, light→dark."""
    p = palette()
    # Sequential ramp: a single hue stepped away from the mode's own surface.
    scale = ([[0, "#eef4fc"], [0.5, "#6aa4e4"], [1, "#12447d"]]
             if p is LIGHT else
             [[0, "#1b2733"], [0.5, "#2f6cb5"], [1, "#8cbcf2"]])
    fmt = ":,.0f" if metric.endswith("rows") else ":.4g"
    vals = piv.values.astype(float)

    # Counts span orders of magnitude, so step the ramp on log10 and label the
    # colorbar with the real values.  Bounded metrics stay linear.
    log_ramp = metric.endswith("rows") and np.nanmax(vals) / max(
        np.nanmin(vals), 1) >= 50
    if log_ramp:
        z = np.log10(np.clip(vals, 1, None))
        ticks = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
        ticks = [t for t in ticks if t <= np.nanmax(vals) * 1.2]
        cbar = dict(tickvals=[np.log10(t) for t in ticks],
                    ticktext=[f"{t:,}" for t in ticks])
    else:
        z = vals
        cbar = {}

    fig = go.Figure(go.Heatmap(
        z=z, x=list(piv.columns), y=list(piv.index), customdata=vals,
        colorscale=scale, hoverongaps=False,
        xgap=2, ygap=2,   # surface gap between cells
        colorbar=dict(title=dict(text=metric, side="right"), outlinewidth=0,
                      tickfont=dict(color=p["muted"]), **cbar),
        hovertemplate=("%{y}<br>%{x}<br>" + metric +
                       ": %{customdata" + fmt + "}<extra></extra>"),
    ))
    fig.update_layout(margin=dict(t=30, b=40, l=10, r=10))
    fig = _style(fig, p, max(420, len(piv) * 17 + 180))
    fig.update_xaxes(side="top", tickangle=-35, gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=p["ink"], size=11))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=p["ink"], size=11))
    return fig


def timeline(sub: pd.DataFrame) -> go.Figure:
    """Collection-date span per facility — a dot per year-quarter with samples."""
    p = palette()
    d = sub.dropna(subset=["collection_dt"])
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="No parseable collection dates in this selection.",
                           showarrow=False, font=dict(size=15, color=p["muted"]))
        return _style(fig, p, 260)

    g = (d.groupby(["facility", d["collection_dt"].dt.to_period("Q")])
           .size().rename("n").reset_index())
    g["period"] = g["collection_dt"].dt.to_timestamp()
    order = sorted(g["facility"].unique())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["period"], y=g["facility"], mode="markers",
        marker=dict(size=np.clip(np.sqrt(g["n"]) * 2.2, 8, 30),
                    color=p["series"][0], opacity=0.6,
                    line=dict(color=p["surface"], width=2)),
        customdata=g["n"],
        hovertemplate="%{y}<br>%{x|%Y Q%q}<br>%{customdata:,} rows<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(title="Sample coverage by quarter",
                      xaxis_title="Collection date",
                      yaxis=dict(categoryorder="array",
                                 categoryarray=order[::-1]),
                      margin=dict(t=60, b=50, l=10, r=20))
    fig = _style(fig, p, max(280, len(order) * 34 + 140))
    fig.update_yaxes(tickfont=dict(color=p["ink"]))
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(sub, links, meta, pads):
    st.subheader("Sessions in this selection")

    if sub.empty:
        st.info("No lab results match the current filters.")
    else:
        summary = (sub.groupby("facility")
                      .agg(rows=("result", "size"),
                           analytes=("analyte_norm", "nunique"),
                           samples=("lab_sample_id", "nunique"),
                           reports=("lab_report_id", "nunique"),
                           pads=("pad_WELL_PAD_ID", "nunique"),
                           first=("collection_dt", "min"),
                           last=("collection_dt", "max"))
                      .reset_index())
        summary["detect_%"] = (
            sub[sub["result_flag"] == "="].groupby("facility").size()
            .reindex(summary["facility"]).fillna(0).values
            / summary["rows"].values * 100
        )
        if not links.empty:
            lc = links.groupby("facility").size()
            summary["approved_links"] = (
                summary["facility"].map(lc).fillna(0).astype(int)
            )
        for c in ("first", "last"):
            summary[c] = summary[c].dt.date.astype(str).replace("NaT", "—")

        st.dataframe(
            summary.sort_values("rows", ascending=False),
            width="stretch", hide_index=True,
            column_config={
                "detect_%": st.column_config.NumberColumn("detect %",
                                                          format="%.1f"),
                "rows": st.column_config.NumberColumn(format="%d"),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                hbar(sub.groupby("facility").size(),
                     "Result rows per facility", "Rows"),
                width="stretch")
        with c2:
            st.plotly_chart(
                hbar(sub.groupby("facility")["analyte_norm"].nunique(),
                     "Distinct analytes per facility", "Analytes"),
                width="stretch")

        st.plotly_chart(timeline(sub), width="stretch")

    with st.expander("Session metadata (all sessions on disk)"):
        st.dataframe(meta, width="stretch", hide_index=True)
        no_out = meta.loc[~meta["has_output"], "session"].tolist()
        if no_out:
            st.caption("No output/ results yet: " + ", ".join(no_out))

    if not pads.empty:
        with st.expander("Candidate pads"):
            st.dataframe(pads.drop(columns=["total_tons", "total_bbls"],
                                   errors="ignore"),
                         width="stretch", hide_index=True)


def tab_analyte(sub, analyte):
    a_sub = sub[sub["analyte_norm"] == analyte]
    n_total = int((sub["analyte_norm"] == analyte).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{len(a_sub):,}", f"of {n_total:,} for analyte",
              delta_color="off")
    c2.metric("Detection freq", detection_freq(a_sub))
    c3.metric("Median (detected)", (
        f"{a_sub[a_sub['result_flag'] == '=']['result_value'].median():.4g}"
        if (a_sub["result_flag"] == "=").any() else "—"
    ))
    c4.metric("Facilities", f"{a_sub['facility'].nunique():,}")

    if a_sub["units"].value_counts().shape[0] > 1:
        st.caption(f"⚠ Multiple units in this selection: {top_units(a_sub)}")

    k1, k2, k3 = st.columns([2, 2, 1])
    chart_type = k1.selectbox("Chart type",
                             ["Histogram", "Box plot", "Strip / dot plot", "CDF"])
    color_by = k2.selectbox("Color by", list(COLOR_DIMS),
                            format_func=lambda s: COLOR_DIMS[s])
    log_scale = k3.checkbox("Log scale", value=True)

    if a_sub.empty:
        st.info("No rows match the current filters.")
    else:
        st.plotly_chart(
            build_figure(a_sub, analyte, log_scale, chart_type, color_by),
            width="stretch",
        )

    with st.expander("Per-facility summary for this analyte"):
        if a_sub.empty:
            st.write("—")
        else:
            det = a_sub[a_sub["result_flag"] == "="]
            tbl = (a_sub.groupby("facility")
                   .agg(rows=("result", "size"),
                        samples=("lab_sample_id", "nunique"))
                   .join(det.groupby("facility")["result_value"]
                         .agg(detected="size", median="median",
                              p95=lambda s: s.quantile(0.95), max="max"))
                   .reset_index())
            tbl["detect_%"] = tbl["detected"].fillna(0) / tbl["rows"] * 100
            st.dataframe(tbl.sort_values("rows", ascending=False),
                         width="stretch", hide_index=True)


def tab_coverage(sub):
    st.subheader("Analyte coverage across facilities")
    if sub.empty:
        st.info("No rows match the current filters.")
        return

    metric = st.radio("Cell value", ["Result rows", "Detected rows",
                                     "Detect %", "Median (detected)"],
                      horizontal=True)
    top_n = st.slider("Analytes shown (most rows first)", 10, 200, 40, step=10)

    keep = sub["analyte_norm"].value_counts().head(top_n).index
    d = sub[sub["analyte_norm"].isin(keep)]
    det = d[d["result_flag"] == "="]

    if metric == "Result rows":
        piv = d.pivot_table(index="analyte_norm", columns="facility",
                            values="result", aggfunc="size")
    elif metric == "Detected rows":
        piv = det.pivot_table(index="analyte_norm", columns="facility",
                              values="result", aggfunc="size")
    elif metric == "Detect %":
        tot = d.pivot_table(index="analyte_norm", columns="facility",
                            values="result", aggfunc="size")
        hit = det.pivot_table(index="analyte_norm", columns="facility",
                              values="result", aggfunc="size")
        piv = (hit.reindex_like(tot).fillna(0) / tot * 100).round(1)
    else:
        piv = det.pivot_table(index="analyte_norm", columns="facility",
                              values="result_value", aggfunc="median")

    piv = piv.reindex(keep).dropna(how="all")
    st.plotly_chart(coverage_heatmap(piv, metric), width="stretch")
    st.caption(
        f"{len(piv):,} analytes × {piv.shape[1]} facilities. "
        "Blank = analyte absent from that facility's curated results."
    )
    with st.expander("Table view"):
        st.dataframe(piv, width="stretch", height=520)
    st.download_button("Download this matrix (CSV)",
                       piv.to_csv().encode("utf-8"),
                       file_name=f"analyte_coverage_{metric.lower().replace(' ', '_')}.csv",
                       mime="text/csv")


def tab_results(sub, analyte):
    st.subheader("Curated lab results")
    if sub.empty:
        st.info("No rows match the current filters.")
        return

    only_analyte = st.checkbox(f"Limit to selected analyte ({analyte})",
                               value=True)
    scoped = sub[sub["analyte_norm"] == analyte] if only_analyte else sub

    limit = st.number_input("Max rows to display", 100, 100_000, 2_000,
                            step=500)
    show = scoped.head(int(limit)).copy()

    cols = ["facility", "analyte_norm", "result", "result_value",
            "result_flag", "units", "matrix_norm", "f26r_waste_code",
            "pad_label", "lab_sample_id", "collection_date", "lab_report_id",
            "lab_name", "client_name", "project_name", "f26r_company",
            "f26r_location"]
    cols = [c for c in cols if c in show.columns]
    tbl = show[cols].reset_index(drop=True)
    tbl.insert(0, "source", pdf_url_col(show, "original_page")
               .reset_index(drop=True))

    st.dataframe(
        tbl, width="stretch", hide_index=True, height=620,
        column_config={"source": st.column_config.LinkColumn(
            "source", display_text="pdf")},
    )
    st.caption(f"Showing {len(tbl):,} of {len(scoped):,} filtered rows.")

    # Serialising the full selection costs ~1s / ~60 MB, so only do it on ask
    # rather than on every rerun.
    if st.checkbox(f"Prepare CSV of all {len(scoped):,} filtered rows"):
        st.download_button("Download filtered rows (CSV)",
                           scoped[cols].to_csv(index=False).encode("utf-8"),
                           file_name="filtered_lab_results.csv",
                           mime="text/csv")


def tab_links(links, facilities):
    st.subheader("Approved document links")
    if links.empty:
        st.info("No approved_links.csv found in the selected sessions.")
        return
    sel = links[links["facility"].isin(facilities)]
    if sel.empty:
        st.info("No approved links for the selected facilities.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved links", f"{len(sel):,}")
    c2.metric("Distinct documents", f"{sel['original_filename'].nunique():,}")
    c3.metric("Distinct pads", f"{sel['pad_WELL_PAD_ID'].nunique():,}")

    st.plotly_chart(hbar(sel.groupby("facility").size(),
                         "Approved links per facility", "Links"),
                    width="stretch")

    cols = [c for c in ["facility", "pad_name", "pad_WELL_PAD_ID",
                        "lab_report_id", "report_id", "source", "confidence",
                        "loc_score", "project_name", "lab_name", "client_name",
                        "set_name", "original_filename", "first_page", "notes"]
            if c in sel.columns]
    tbl = sel[cols].reset_index(drop=True)
    tbl.insert(0, "pdf", pdf_url_col(sel, "first_page").reset_index(drop=True))

    st.dataframe(tbl, width="stretch", hide_index=True, height=560,
                 column_config={"pdf": st.column_config.LinkColumn(
                     "pdf", display_text="open")})
    st.download_button("Download links (CSV)",
                       sel[cols].to_csv(index=False).encode("utf-8"),
                       file_name="approved_links.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.title("🏭 Facility Session Explorer")

    if not SESSIONS_DIR.is_dir():
        st.error(f"Sessions directory not found: {SESSIONS_DIR}")
        return

    with st.spinner("Loading sessions…"):
        results, links, pads, meta = load_sessions(session_signature())

    if results.empty:
        st.warning("No session has an output/lab_results.parquet yet. "
                   "Run the Output step in facility_curator.py first.")
        st.dataframe(meta, width="stretch", hide_index=True)
        return

    # -- Sidebar filters ---------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        fac_counts = results["facility"].value_counts()
        fac_options = list(fac_counts.index)
        facilities = st.multiselect(
            "Facility", options=fac_options, default=fac_options,
            format_func=lambda f: f"{f}  ({fac_counts[f]:,})",
        )
        if not facilities:
            st.warning("Select at least one facility.")
            st.stop()

        base = results[results["facility"].isin(facilities)]
        st.caption(f"{len(base):,} result rows in scope")

        st.divider()
        all_flags = sorted(base["result_flag"].dropna().unique())
        flags = st.multiselect(
            "Result flag", options=all_flags,
            default=[f for f in all_flags if f in NUMERIC_FLAGS],
            format_func=lambda f: FLAG_LABELS.get(f, f),
        )

        st.divider()
        matrices = sorted(base["matrix_norm"].dropna().unique())
        sel_matrices = st.multiselect("Matrix", options=matrices,
                                      default=matrices)

        st.divider()
        wc_counts = base["f26r_waste_code"].value_counts()
        sel_wcs = st.multiselect(
            "Waste code (f26r)", options=list(wc_counts.index),
            default=list(wc_counts.index),
            format_func=lambda c: f"{c}  ({wc_counts[c]:,})",
        )

        st.divider()
        pad_counts = base["pad_label"].value_counts()
        sel_pads = st.multiselect(
            "Well pad", options=list(pad_counts.index),
            default=[],
            format_func=lambda p: f"{p}  ({pad_counts[p]:,})",
            help="Empty = all pads.",
        )

        st.divider()
        dates = base["collection_dt"].dropna()
        use_dates = False
        if not dates.empty:
            lo, hi = dates.min().date(), dates.max().date()
            if lo < hi:
                use_dates = st.checkbox("Filter by collection date")
                if use_dates:
                    dr = st.date_input("Collection date range", (lo, hi),
                                       min_value=lo, max_value=hi)
                    if not isinstance(dr, (tuple, list)) or len(dr) != 2:
                        use_dates = False

    # -- Apply filters -----------------------------------------------------
    sub = base[
        base["result_flag"].isin(flags)
        & base["matrix_norm"].isin(sel_matrices)
        & base["f26r_waste_code"].isin(sel_wcs)
    ]
    if sel_pads:
        sub = sub[sub["pad_label"].isin(sel_pads)]
    if use_dates:
        d0 = pd.Timestamp(dr[0])
        d1 = pd.Timestamp(dr[1]) + pd.Timedelta(days=1)
        sub = sub[sub["collection_dt"].between(d0, d1, inclusive="left")]

    # -- Headline ----------------------------------------------------------
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Facilities", f"{sub['facility'].nunique():,}")
    h2.metric("Result rows", f"{len(sub):,}", f"of {len(results):,} total",
              delta_color="off")
    h3.metric("Analytes", f"{sub['analyte_norm'].nunique():,}")
    h4.metric("Samples", f"{sub['lab_sample_id'].nunique():,}")
    h5.metric("Detection freq", detection_freq(sub))

    # -- Analyte selector (scoped to the filtered rows) --------------------
    a_counts = sub["analyte_norm"].value_counts()
    if a_counts.empty:
        a_counts = base["analyte_norm"].value_counts()
    a_labels = [f"{n}  ({c:,})" for n, c in a_counts.items()]
    a_map = dict(zip(a_labels, a_counts.index))
    default_idx = next((i for i, n in enumerate(a_counts.index)
                        if n == "Benzene"), 0)
    analyte = a_map[st.selectbox("Analyte", a_labels, index=default_idx)]

    t1, t2, t3, t4, t5 = st.tabs(
        ["Overview", "Analyte", "Coverage", "Results", "Links"])
    with t1:
        tab_overview(sub, links[links["facility"].isin(facilities)]
                     if not links.empty else links, meta, pads)
    with t2:
        tab_analyte(sub, analyte)
    with t3:
        tab_coverage(sub)
    with t4:
        tab_results(sub, analyte)
    with t5:
        tab_links(links, facilities)


if __name__ == "__main__":
    main()

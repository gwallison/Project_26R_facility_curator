"""
generate_facility_report.py

Produce a PDF summary of curated lab results for a single facility_curator session.
Report is broken into sections by matrix group (Solid, Wastewater, Water, Leachate, Other).
Histograms within each section are colored by f26r_waste_code.

Usage:
    python generate_facility_report.py <session-slug>
    python generate_facility_report.py westmoreland-waste-llc-sanitary-landfill

Output:
    sessions/<slug>/output/facility_report_YYYYMMDD.pdf
"""

import sys
import pathlib
import datetime
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
SESSIONS_ROOT = pathlib.Path(__file__).parent / "sessions"

# ── constants ─────────────────────────────────────────────────────────────────
NUMERIC_FLAGS = {"=", "<", ">", "~"}
TOP_N_ANALYTES = 30
MIN_ROWS_FOR_SECTION = 50
UNIT_SPLIT_MIN_FRAC = 0.10

# Always included regardless of rank, if present in the section data
PINNED_ANALYTES = ["Gross Alpha", "Gross Beta", "Radium-226", "Radium-228"]

# raw matrix string (lowercased) → normalized group
_MATRIX_MAP = {
    # Solid
    "solid": "Solid",
    "solid/grab": "Solid",
    "solid/composite": "Solid",
    "(solid/grab)": "Solid",
    "soil": "Solid",
    "tenorm solids": "Solid",
    "sludge": "Solid",
    # Wastewater
    "waste water": "Wastewater",
    "wastewater": "Wastewater",
    "waste - liquid": "Wastewater",
    "liqwaste": "Wastewater",
    "waste liquid": "Wastewater",
    # Water
    "water": "Water",
    "water/grab": "Water",
    "water/composite": "Water",
    "aqueous": "Water",
    "non-potable water": "Water",
    "non potable water": "Water",
    "pws": "Water",
    # Leachate
    "leachate": "Leachate",
    "tclp leach": "Leachate",
    # Other
    "other": "Other",
    "other/grab": "Other",
    "(other/grab)": "Other",
    "oil/other": "Other",
}

MATRIX_ORDER = ["Solid", "Wastewater", "Water", "Leachate", "Other"]
MATRIX_COLORS = {
    "Solid":      "#8c564b",
    "Wastewater": "#1f77b4",
    "Water":      "#2ca02c",
    "Leachate":   "#9467bd",
    "Other":      "#7f7f7f",
}

# tab10-style palette for waste codes (assigned dynamically)
_WC_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ── normalization ─────────────────────────────────────────────────────────────
def normalize_matrix(s: pd.Series) -> pd.Series:
    lowered = s.fillna("").str.strip().str.lower()
    return lowered.map(_MATRIX_MAP).fillna("Other")


def normalize_unit(u) -> str:
    if pd.isna(u) or str(u).strip() == "":
        return "(none)"
    s = str(u).strip()
    s = s.replace("�", "u").replace("µ", "u").replace("μ", "u")
    sl = s.lower()
    if sl in ("mg/l", "mg/1"):
        return "mg/L"
    if sl in ("ug/l", "ug/1"):
        return "ug/L"
    if sl == "ng/l":
        return "ng/L"
    if "caco3" in sl:
        return "mg/L as CaCO3"
    if "as n" in sl:
        return "mg/L as N"
    if sl in ("pci/l",):
        return "pCi/L"
    if sl in ("pci/g", "pci/gr"):
        return "pCi/g"
    if "mhos/cm" in sl:
        return "umhos/cm"
    if sl in ("su", "s.u.", "std. units", "standard units", "std units", "ph units"):
        return "S.U."
    if sl in ("mg/kg dry", "mg/kg"):
        return "mg/kg"
    return s


# ── data loading ──────────────────────────────────────────────────────────────
def load_session(slug: str) -> tuple[pd.DataFrame, dict]:
    path = SESSIONS_ROOT / slug / "output" / "lab_results.parquet"
    if not path.exists():
        print(f"ERROR: no lab_results.parquet found for session '{slug}'")
        print(f"  Expected: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    df["received_date"] = pd.to_datetime(df["received_date"], errors="coerce")
    df["matrix_norm"] = normalize_matrix(df["matrix"])
    df["units_norm"] = df["units"].apply(normalize_unit)
    df["f26r_waste_code"] = df["f26r_waste_code"].fillna("(unassigned)")

    meta = {
        "slug": slug,
        "n_rows": len(df),
        "n_pads": df["pad_WELL_PAD_ID"].nunique(),
        "n_reports": df["lab_report_id"].nunique(),
        "n_files": df["original_filename"].nunique(),
        "date_min": df["received_date"].min(),
        "date_max": df["received_date"].max(),
        "waste_codes": df["f26r_waste_code"].value_counts().to_dict(),
        "matrix_counts": df["matrix_norm"].value_counts().to_dict(),
    }
    return df, meta


# ── analyte entry building ────────────────────────────────────────────────────
def build_entries(df: pd.DataFrame) -> list[dict]:
    """Top-N analytes by row count; split by unit if multiple units are significant.
    PINNED_ANALYTES are appended after the top-N if not already present."""
    top = list(df["analyte_norm"].value_counts().head(TOP_N_ANALYTES).index)
    for analyte in PINNED_ANALYTES:
        if analyte not in top and (df["analyte_norm"] == analyte).any():
            top.append(analyte)
    entries = []
    for analyte in top:
        sub = df[df["analyte_norm"] == analyte]
        total = len(sub)
        uc = sub["units_norm"].value_counts()
        sig = [(u, c) for u, c in uc.items() if c / total >= UNIT_SPLIT_MIN_FRAC]
        if len(sig) > 1:
            for u, c in sig:
                entries.append(dict(display=f"{analyte}  [{u}]", analyte=analyte,
                                    units_norm=u, total=c))
        else:
            entries.append(dict(display=analyte, analyte=analyte,
                                units_norm=None, total=total))
    return entries


def get_subset(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    sub = df[df["analyte_norm"] == entry["analyte"]]
    if entry["units_norm"] is not None:
        sub = sub[sub["units_norm"] == entry["units_norm"]]
    return sub


def compute_stats(sub: pd.DataFrame) -> dict:
    n_total = len(sub)
    numeric = sub[sub["result_flag"].isin(NUMERIC_FLAGS)]
    detected = numeric[numeric["result_flag"] == "="]["result_value"].dropna()
    n_det = len(detected)
    det_freq = f"{n_det / n_total * 100:.1f}%  ({n_det:,} / {n_total:,})" if n_total else "—"
    median = f"{detected.median():.4g}" if len(detected) else "—"
    p90 = f"{detected.quantile(0.90):.4g}" if len(detected) else "—"
    units = sub["units_norm"].value_counts().index[0] if sub["units_norm"].notna().any() else "—"
    return dict(n_total=n_total, n_det=n_det, det_freq=det_freq,
                median=median, p90=p90, units=units)


# ── PDF page builders ─────────────────────────────────────────────────────────
def fig_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_title_page(pdf: PdfPages, meta: dict):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    title = meta["slug"].replace("-", " ").title()
    ax.text(0.5, 0.93, "Facility Lab Results — Internal Summary",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.88, textwrap.fill(title, 55),
            ha="center", va="top", fontsize=12, color="#333333",
            transform=ax.transAxes)
    ax.text(0.5, 0.83, datetime.date.today().strftime("Generated %B %d, %Y"),
            ha="center", va="top", fontsize=10, color="#777777",
            transform=ax.transAxes)

    date_range = (
        f"{meta['date_min'].strftime('%Y-%m-%d')} — {meta['date_max'].strftime('%Y-%m-%d')}"
        if pd.notna(meta["date_min"]) else "unknown"
    )
    overview = [
        f"Total rows:               {meta['n_rows']:>10,}",
        f"Unique well pads:         {meta['n_pads']:>10,}",
        f"Unique lab reports:       {meta['n_reports']:>10,}",
        f"Unique source files:      {meta['n_files']:>10,}",
        f"Date range:               {date_range}",
    ]
    ax.text(0.10, 0.77, "\n".join(overview), ha="left", va="top", fontsize=10,
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#aaaaaa"))

    # Matrix breakdown (left column)
    ax.text(0.10, 0.60, "Rows by matrix group:", ha="left", va="top",
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    y = 0.56
    for mx in MATRIX_ORDER:
        n = meta["matrix_counts"].get(mx, 0)
        if n:
            color = MATRIX_COLORS.get(mx, "#555555")
            ax.add_patch(plt.Rectangle((0.10, y - 0.012), 0.012, 0.018,
                                       transform=ax.transAxes, color=color))
            ax.text(0.13, y, f"{mx:<14} {n:>8,}", ha="left", va="top",
                    fontsize=9, family="monospace", transform=ax.transAxes)
            y -= 0.038

    # Waste code breakdown (right column)
    ax.text(0.55, 0.60, "Rows by waste code:", ha="left", va="top",
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    y = 0.56
    for wc, n in sorted(meta["waste_codes"].items(), key=lambda x: -x[1])[:12]:
        ax.text(0.55, y, f"{str(wc):<16} {n:>8,}", ha="left", va="top",
                fontsize=9, family="monospace", transform=ax.transAxes)
        y -= 0.038

    ax.text(0.5, 0.07,
            f"Sections: {', '.join(MATRIX_ORDER)}  "
            f"(sections with < {MIN_ROWS_FOR_SECTION} rows omitted)\n"
            "Histograms: log₁₀ scale, colored by waste code.  "
            "Median and 90th pct computed on detected (=) values only.",
            ha="center", va="bottom", fontsize=8, color="#777777",
            transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_section_header(pdf: PdfPages, matrix_name: str, df_mx: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    color = MATRIX_COLORS.get(matrix_name, "#555555")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                transform=ax.transAxes, color=color, clip_on=False))
    ax.text(0.5, 0.94, f"Matrix: {matrix_name}",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)

    n_numeric = df_mx["result_flag"].isin(NUMERIC_FLAGS).sum()
    n_det = (df_mx["result_flag"] == "=").sum()
    stats = [
        f"Rows in this section:     {len(df_mx):>8,}",
        f"  Numeric (=, <, >, ~):   {n_numeric:>8,}",
        f"  Detected (=):           {n_det:>8,}",
        f"Unique analyte_norm:      {df_mx['analyte_norm'].nunique():>8,}",
        f"Unique well pads:         {df_mx['pad_WELL_PAD_ID'].nunique():>8,}",
        f"Unique lab reports:       {df_mx['lab_report_id'].nunique():>8,}",
    ]
    ax.text(0.10, 0.82, "\n".join(stats), ha="left", va="top", fontsize=10,
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#aaaaaa"))

    # Waste code breakdown
    ax.text(0.10, 0.57, "Waste code breakdown:", ha="left", va="top",
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    y = 0.53
    for code, count in df_mx["f26r_waste_code"].value_counts().items():
        ax.text(0.13, y, f"{str(code):<16} {count:>8,}", ha="left", va="top",
                fontsize=9, family="monospace", transform=ax.transAxes)
        y -= 0.035

    # Raw matrix values
    ax.text(0.55, 0.57, "Raw matrix values:", ha="left", va="top",
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    y = 0.53
    for raw_mx, count in df_mx["matrix"].value_counts().head(10).items():
        ax.text(0.55, y, f"{str(raw_mx):<20} {count:>6,}", ha="left", va="top",
                fontsize=9, family="monospace", transform=ax.transAxes)
        y -= 0.035

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(pdf: PdfPages, entries: list[dict], df_mx: pd.DataFrame,
                       matrix_name: str):
    rows_per_page = 32
    header = ["Analyte  [units]", "Rows", "Det. freq", "Median\n(det.)", "90th pct\n(det.)", "Units"]
    col_widths = [0.38, 0.07, 0.18, 0.12, 0.12, 0.13]

    all_rows = []
    for e in entries:
        sub = get_subset(df_mx, e)
        s = compute_stats(sub)
        display = textwrap.shorten(e["display"], width=48, placeholder="…")
        all_rows.append([display, f"{s['n_total']:,}", s["det_freq"],
                         s["median"], s["p90"],
                         e["units_norm"] if e["units_norm"] else s["units"]])

    chunks = [all_rows[i:i + rows_per_page] for i in range(0, len(all_rows), rows_per_page)]
    for ci, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        label = f"{matrix_name} — Summary Table"
        if len(chunks) > 1:
            label += f"  (page {ci + 1} of {len(chunks)})"
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)

        tbl = ax.table(cellText=chunk, colLabels=header, colWidths=col_widths,
                       loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.4)

        hdr_color = MATRIX_COLORS.get(matrix_name, "#2c5f8a")
        for j in range(len(header)):
            cell = tbl[(0, j)]
            cell.set_facecolor(hdr_color)
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(chunk)):
            for j in range(len(header)):
                cell = tbl[(i + 1, j)]
                cell.set_facecolor("#f0f6fb" if i % 2 == 0 else "white")
                cell.set_edgecolor("#dddddd")

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _build_wc_colors(df_mx: pd.DataFrame) -> tuple[dict, list]:
    """Assign colors to waste codes; (unassigned) always last."""
    wc_present = sorted(
        df_mx["f26r_waste_code"].unique(),
        key=lambda x: (x == "(unassigned)", x),
    )
    wc_colors = {wc: _WC_PALETTE[i % len(_WC_PALETTE)] for i, wc in enumerate(wc_present)}
    return wc_colors, wc_present


def make_histogram(ax, sub: pd.DataFrame, entry: dict, stats: dict,
                   wc_colors: dict, wc_order: list):
    numeric = sub[sub["result_flag"].isin(NUMERIC_FLAGS)].copy()
    numeric = numeric[numeric["result_value"].notna() & (numeric["result_value"] > 0)]

    if numeric.empty:
        ax.text(0.5, 0.5, "No positive numeric values", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(entry["display"], fontsize=9, fontweight="bold")
        return

    log_vals = np.log10(numeric["result_value"].clip(lower=1e-15))
    lo, hi = log_vals.quantile(0.005), log_vals.quantile(0.995)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    bins = np.linspace(lo, hi, 40)

    for wc in wc_order:
        grp = numeric[numeric["f26r_waste_code"] == wc]
        if grp.empty:
            continue
        v = np.log10(grp["result_value"].clip(lower=1e-15))
        ax.hist(v, bins=bins, alpha=0.65, color=wc_colors[wc], label=wc, edgecolor="none")

    detected = numeric[numeric["result_flag"] == "="]["result_value"].dropna()
    detected = detected[detected > 0]
    if len(detected):
        med = np.log10(detected.median())
        p90v = np.log10(detected.quantile(0.90))
        ax.axvline(med, color="#333333", lw=1.2, ls="--",
                   label=f"Median ({detected.median():.3g})")
        ax.axvline(p90v, color="#333333", lw=1.0, ls=":",
                   label=f"90th pct ({detected.quantile(0.90):.3g})")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$10^{{{x:.0f}}}$"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel(f"log₁₀  [{stats['units']}]", fontsize=7.5)
    ax.set_ylabel("Count", fontsize=7.5)
    ax.tick_params(labelsize=7)

    ann = (f"n={stats['n_total']:,}   det={stats['det_freq']}\n"
           f"median={stats['median']}   p90={stats['p90']}")
    ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                      edgecolor="#cccccc"))
    ax.set_title(textwrap.shorten(entry["display"], width=52, placeholder="…"),
                 fontsize=9, fontweight="bold", pad=4)


def make_histogram_pages(pdf: PdfPages, entries: list[dict], df_mx: pd.DataFrame,
                         matrix_name: str):
    wc_colors, wc_order = _build_wc_colors(df_mx)

    legend_handles = [
        Patch(facecolor=wc_colors[wc], alpha=0.7, label=f"Waste code {wc}")
        for wc in wc_order
    ] + [
        Line2D([0], [0], color="#333333", lw=1.2, ls="--", label="Median (detected)"),
        Line2D([0], [0], color="#333333", lw=1.0, ls=":", label="90th pct (detected)"),
    ]

    pairs = [entries[i:i + 2] for i in range(0, len(entries), 2)]
    for pair in pairs:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        fig.subplots_adjust(hspace=0.50, top=0.93, bottom=0.12)

        for i, entry in enumerate(pair):
            sub = get_subset(df_mx, entry)
            stats = compute_stats(sub)
            make_histogram(axes[i], sub, entry, stats, wc_colors, wc_order)

        if len(pair) == 1:
            axes[1].axis("off")

        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 5),
                   fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_facility_report.py <session-slug>\n")
        print("Available sessions:")
        for p in sorted(SESSIONS_ROOT.iterdir()):
            out = p / "output" / "lab_results.parquet"
            marker = "+" if out.exists() else "-"
            print(f"  [{marker}]  {p.name}")
        sys.exit(1)

    slug = sys.argv[1]
    out_path = (SESSIONS_ROOT / slug / "output"
                / f"facility_report_{datetime.date.today().strftime('%Y%m%d')}.pdf")

    fig_style()
    print(f"Loading session: {slug}")
    df, meta = load_session(slug)
    print(f"  {meta['n_rows']:,} rows  |  {meta['n_pads']} pads  "
          f"|  {meta['n_reports']} lab reports  |  {meta['n_files']} files")

    with PdfPages(out_path) as pdf:
        d = pdf.infodict()
        d["Title"] = f"Facility Lab Results — {slug}"
        d["Author"] = "facility_curator / generate_facility_report.py"
        d["CreationDate"] = datetime.datetime.now()

        print("Writing title page…")
        make_title_page(pdf, meta)

        for matrix_name in MATRIX_ORDER:
            df_mx = df[df["matrix_norm"] == matrix_name].copy()
            if len(df_mx) < MIN_ROWS_FOR_SECTION:
                print(f"  Skipping {matrix_name}: {len(df_mx)} rows (< {MIN_ROWS_FOR_SECTION})")
                continue

            print(f"\nSection: {matrix_name}  ({len(df_mx):,} rows)")
            make_section_header(pdf, matrix_name, df_mx)

            entries = build_entries(df_mx)
            print(f"  {len(entries)} analyte entries")
            make_summary_table(pdf, entries, df_mx, matrix_name)
            make_histogram_pages(pdf, entries, df_mx, matrix_name)

    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()

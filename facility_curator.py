"""
facility_curator.py
Curation workflow for building facility-specific 26R lab result datasets.

Creates a persistent session per target waste facility.  Workflow:
  Session        — search facility name, create/load session
  Candidate Pads — review/edit the pad list (include/exclude)
  Candidate Links— auto-linked + manual links; find unlinked pads
  Review         — step through candidates, approve / reject
  Output         — compile approved_links.csv + lab_results.parquet

Usage (from this directory):
    streamlit run facility_curator.py
"""

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

LINKS_PARQUET = Path(
    r"G:\My Drive\sandbox\26R\link_DEP_waste_to_labs\data\links.parquet"
)
PAD_PARQUET = Path(
    r"G:\My Drive\sandbox\26R\link_DEP_waste_to_labs\data\pad_vocab.parquet"
)
LAB_PARQUET = Path(
    r"G:\My Drive\sandbox\26R\data_cleanup_26R\processed\lab_results_result_parsed.parquet"
)
WASTE_PARQUET = Path(
    r"G:\My Drive\Info_home\Projects\Project_Homes\Produced water"
    r"\PA_DEP_waste_dataset\pa_waste_merged.parquet"
)

GCS_BASE = "https://storage.googleapis.com/fta-form26r-library"

_WASTE_COLS = [
    "UNCONVENTIONAL_IND", "pad_WELL_PAD_ID", "PERMIT_NUM", "WELL_NO",
    "_period_label", "PRODUCT_TYPE", "QUANTITY", "UNITS",
    "DISPOSAL_METHOD", "WASTE_FACILITY_NAME",
]

# Columns persisted in candidate_links.csv
LINK_COLS = [
    "original_filename", "lab_report_id", "pad_WELL_PAD_ID",
    "source", "status", "notes",
    "confidence", "loc_score", "project_name",
    "lab_name", "client_name", "set_name", "first_page",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"\s+", "-", s.strip())[:50] or "unnamed"


def pdf_url(row) -> str:
    sn = quote(str(row.get("set_name", "")), safe="")
    fn = quote(str(row["original_filename"]), safe="")
    try:
        pg = int(row["first_page"]) if pd.notna(row.get("first_page")) else 1
    except (ValueError, TypeError):
        pg = 1
    return f"{GCS_BASE}/full-set/{sn}/{fn}#page={pg}"


def status_icon(s: str) -> str:
    return {"pending": "⏳", "approved": "✅", "rejected": "❌"}.get(
        str(s).lower(), "?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached global data (read-only)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _load_links():
    df = pd.read_parquet(LINKS_PARQUET)
    return (
        df[df["confidence"] != "no_match"].copy(),
        df[df["confidence"] == "no_match"].copy(),
    )


@st.cache_data
def _load_pads():
    return pd.read_parquet(PAD_PARQUET)


@st.cache_data
def _load_waste():
    df = pd.read_parquet(WASTE_PARQUET, columns=_WASTE_COLS)
    return (
        df[df["UNCONVENTIONAL_IND"] == "Yes"]
        .drop(columns="UNCONVENTIONAL_IND")
        .reset_index(drop=True)
    )


@st.cache_data
def _load_lab_results():
    return pd.read_parquet(LAB_PARQUET)


# ─────────────────────────────────────────────────────────────────────────────
# Session I/O
# ─────────────────────────────────────────────────────────────────────────────
def list_sessions():
    out = []
    for d in sorted(SESSIONS_DIR.iterdir()):
        j = d / "session.json"
        if d.is_dir() and j.exists():
            with open(j) as f:
                m = json.load(f)
            m["_path"] = d
            out.append(m)
    return out


def _touch(sp: Path):
    j = sp / "session.json"
    with open(j) as f:
        m = json.load(f)
    m["last_modified"] = str(date.today())
    with open(j, "w") as f:
        json.dump(m, f, indent=2)


def load_cpads(sp: Path) -> pd.DataFrame:
    p = sp / "candidate_pads.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["included"] = df["included"].map(
        {"True": True, "False": False, True: True, False: False}
    ).fillna(True).astype(bool)
    # backfill columns added after a session was first created
    for col in ("total_tons", "total_bbls"):
        if col not in df.columns:
            df[col] = float("nan")
    return df


def save_cpads(sp: Path, df: pd.DataFrame):
    df.to_csv(sp / "candidate_pads.csv", index=False)
    _touch(sp)


def load_clinks(sp: Path) -> pd.DataFrame:
    p = sp / "candidate_links.csv"
    if not p.exists():
        return pd.DataFrame(columns=LINK_COLS)
    df = pd.read_csv(p, dtype={"lab_report_id": str, "notes": str})
    df["notes"] = df["notes"].fillna("")
    for c in LINK_COLS:
        if c not in df.columns:
            df[c] = None
    return df[LINK_COLS].copy()


def save_clinks(sp: Path, df: pd.DataFrame):
    df[LINK_COLS].to_csv(sp / "candidate_links.csv", index=False)
    _touch(sp)


def _auto_links_df(matched: pd.DataFrame, pad_ids: set) -> pd.DataFrame:
    rows = matched[matched["pad_WELL_PAD_ID"].isin(pad_ids)].copy()
    if rows.empty:
        return pd.DataFrame(columns=LINK_COLS)
    cl = pd.DataFrame(columns=LINK_COLS)
    for c in LINK_COLS:
        if c == "source":
            cl[c] = "auto"
        elif c == "status":
            cl[c] = "pending"
        elif c == "notes":
            cl[c] = ""
        elif c in rows.columns:
            cl[c] = rows[c].values
        else:
            cl[c] = None
    return cl.reset_index(drop=True)


def create_session(
    facility_name: str,
    facility_query: str,
    pad_sum: pd.DataFrame,
    matched: pd.DataFrame,
) -> Path:
    slug = slugify(facility_name)
    sp = SESSIONS_DIR / slug
    if sp.exists():
        for i in range(2, 100):
            sp2 = SESSIONS_DIR / f"{slug}-{i}"
            if not sp2.exists():
                sp = sp2
                break
    sp.mkdir(parents=True)
    (sp / "output").mkdir()

    with open(sp / "session.json", "w") as f:
        json.dump(
            {
                "facility_name":  facility_name,
                "facility_query": facility_query,
                "created":        str(date.today()),
                "last_modified":  str(date.today()),
            },
            f,
            indent=2,
        )

    cp = pad_sum[[
        "pad_WELL_PAD_ID", "pad_WELL_PAD", "CLIENT", "COUNTY",
        "n_waste_records", "total_tons", "total_bbls", "first_period", "last_period",
    ]].copy()
    cp["included"] = False
    save_cpads(sp, cp)

    save_clinks(sp, pd.DataFrame(columns=LINK_COLS))
    return sp


def sync_auto_links(sp: Path, matched: pd.DataFrame, pad_ids: set | None = None) -> int:
    """Add auto-links for the given pad_ids (or all included pads if None); never removes rows."""
    cp = load_cpads(sp)
    cl = load_clinks(sp)
    if pad_ids is None:
        pad_ids = set(cp.loc[cp["included"], "pad_WELL_PAD_ID"].dropna())
    included = pad_ids
    # Normalise all three key parts to str to avoid int/float type mismatches
    existing = set(
        zip(
            cl["original_filename"].astype(str),
            cl["lab_report_id"].astype(str),
            cl["pad_WELL_PAD_ID"].astype(str),
        )
    )
    included_str = set(str(v) for v in included)
    new_rows = []
    for _, r in matched[matched["pad_WELL_PAD_ID"].astype(str).isin(included_str)].iterrows():
        key = (str(r["original_filename"]), str(r.get("lab_report_id", "")), str(r["pad_WELL_PAD_ID"]))
        if key not in existing:
            row = {c: r.get(c) for c in LINK_COLS}
            row.update(source="auto", status="pending", notes="")
            new_rows.append(row)
    if new_rows:
        cl = pd.concat(
            [cl, pd.DataFrame(new_rows, columns=LINK_COLS)],
            ignore_index=True,
        )
        # Belt-and-suspenders dedup — preserves first occurrence (keeps status/notes)
        cl = cl.drop_duplicates(
            subset=["original_filename", "lab_report_id", "pad_WELL_PAD_ID"],
            keep="first",
        )
        save_clinks(sp, cl)
    return len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="26R Facility Curator", layout="wide")
st.title("26R Facility Curator")

matched_links, no_match_labs = _load_links()
all_pads = _load_pads()
waste = _load_waste()

if "session_path" not in st.session_state:
    st.session_state["session_path"] = None

sp: Path | None = st.session_state["session_path"]
session_meta: dict = {}
if sp and (sp / "session.json").exists():
    with open(sp / "session.json") as f:
        session_meta = json.load(f)

# Sidebar: session indicator
with st.sidebar:
    if sp:
        st.success(f"**Session:**\n{session_meta.get('facility_name', sp.name)}")
        st.caption(f"Modified: {session_meta.get('last_modified', '—')}")
        if st.button("Close session"):
            st.session_state["session_path"] = None
            st.rerun()
    else:
        st.info("No session loaded.\nUse the **Session** tab.")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_session, tab_pads, tab_links, tab_review, tab_output = st.tabs([
    "Session", "Candidate Pads", "Candidate Links", "Review", "Output",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Session
# ══════════════════════════════════════════════════════════════════════════════
with tab_session:
    sessions = list_sessions()
    if sessions:
        st.subheader(f"Existing sessions  ({len(sessions)})")
        for m in sessions:
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.write(f"**{m['facility_name']}**")
            with c2:
                st.caption(f"Modified: {m.get('last_modified', '—')}  ·  created: {m.get('created', '—')}")
            with c3:
                if st.button("Load", key=f"load_{m['_path']}"):
                    st.session_state["session_path"] = Path(m["_path"])
                    st.rerun()
        st.divider()

    st.subheader("Create new session")
    fac_q = st.text_input(
        "Facility name search (substring, case-insensitive)",
        placeholder="e.g. Westmoreland, Eureka, Hart Resource",
        key="new_fac_q",
    )

    if not fac_q.strip():
        st.info("Enter a facility name to search the waste dataset.")
    else:
        hit = waste[
            waste["WASTE_FACILITY_NAME"].str.contains(fac_q.strip(), case=False, na=False)
        ]
        if hit.empty:
            st.warning(f"No waste records match '{fac_q}'.")
        else:
            variants = sorted(hit["WASTE_FACILITY_NAME"].dropna().unique())
            with st.expander(f"{len(variants)} name variant(s) matched"):
                for v in variants:
                    st.write(v)

            tons_by_pad = (
                hit[hit["UNITS"] == "Tons"]
                .groupby("pad_WELL_PAD_ID")["QUANTITY"]
                .sum()
                .rename("total_tons")
            )
            bbls_by_pad = (
                hit[hit["UNITS"] == "Bbls"]
                .groupby("pad_WELL_PAD_ID")["QUANTITY"]
                .sum()
                .rename("total_bbls")
            )
            pad_sum = (
                hit.dropna(subset=["pad_WELL_PAD_ID"])
                .groupby("pad_WELL_PAD_ID")
                .agg(
                    n_waste_records=("QUANTITY", "count"),
                    first_period=("_period_label", "min"),
                    last_period=("_period_label", "max"),
                )
                .reset_index()
                .join(tons_by_pad, on="pad_WELL_PAD_ID", how="left")
                .join(bbls_by_pad, on="pad_WELL_PAD_ID", how="left")
            )
            pad_meta = all_pads[["pad_WELL_PAD_ID", "pad_WELL_PAD", "CLIENT", "COUNTY"]].copy()
            pad_sum = pad_sum.merge(pad_meta, on="pad_WELL_PAD_ID", how="left")
            pad_sum = pad_sum.sort_values("n_waste_records", ascending=False).reset_index(drop=True)

            n_auto = matched_links[
                matched_links["pad_WELL_PAD_ID"].isin(set(pad_sum["pad_WELL_PAD_ID"]))
            ].shape[0]
            st.write(
                f"**{len(pad_sum)} well pads** found  ·  "
                f"**{n_auto}** auto-linked lab reports will be pre-loaded as candidates."
            )

            st.dataframe(
                pad_sum[[
                    "pad_WELL_PAD", "CLIENT", "COUNTY",
                    "n_waste_records", "total_tons", "total_bbls",
                    "first_period", "last_period",
                ]],
                use_container_width=True,
                height=250,
                column_config={
                    "pad_WELL_PAD":    st.column_config.TextColumn("Pad Name",    width="medium"),
                    "CLIENT":          st.column_config.TextColumn("Operator",    width="medium"),
                    "COUNTY":          st.column_config.TextColumn("County",      width="small"),
                    "n_waste_records": st.column_config.NumberColumn("Waste Recs", format="%d",   width="small"),
                    "total_tons":      st.column_config.NumberColumn("Tons",       format="%.1f", width="small"),
                    "total_bbls":      st.column_config.NumberColumn("Bbls",       format="%.1f", width="small"),
                    "first_period":    st.column_config.TextColumn("First Period", width="small"),
                    "last_period":     st.column_config.TextColumn("Last Period",  width="small"),
                },
            )

            default_name = max(variants, key=len)
            facility_name_input = st.text_input(
                "Session name (editable)", value=default_name, key="new_fac_name"
            )

            if st.button("Create session", type="primary", key="btn_create"):
                with st.spinner("Creating session…"):
                    new_sp = create_session(
                        facility_name_input, fac_q.strip(), pad_sum, matched_links
                    )
                st.session_state["session_path"] = new_sp
                st.success(f"Session created: {new_sp.name}")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Candidate Pads
# ══════════════════════════════════════════════════════════════════════════════
with tab_pads:
    if not sp:
        st.info("Load or create a session in the **Session** tab first.")
    else:
        st.subheader(f"Candidate Pads — {session_meta.get('facility_name', '')}")
        cp = load_cpads(sp)
        cl = load_clinks(sp)

        # Merge link counts into display
        link_counts = (
            cl.groupby("pad_WELL_PAD_ID")
            .agg(
                n_candidates=("original_filename", "count"),
                n_approved=("status", lambda x: (x == "approved").sum()),
                n_pending=("status", lambda x: (x == "pending").sum()),
            )
            .reset_index()
        )
        cp_disp = cp.merge(link_counts, on="pad_WELL_PAD_ID", how="left")
        for c in ("n_candidates", "n_approved", "n_pending"):
            cp_disp[c] = cp_disp[c].fillna(0).astype(int)

        edited = st.data_editor(
            cp_disp[[
                "included", "pad_WELL_PAD", "CLIENT", "COUNTY",
                "n_waste_records", "total_tons", "total_bbls",
                "n_candidates", "n_approved", "n_pending",
            ]],
            use_container_width=True,
            height=450,
            disabled=[
                "pad_WELL_PAD", "CLIENT", "COUNTY",
                "n_waste_records", "total_tons", "total_bbls",
                "n_candidates", "n_approved", "n_pending",
            ],
            column_config={
                "included":        st.column_config.CheckboxColumn("Include",     width="small"),
                "pad_WELL_PAD":    st.column_config.TextColumn("Pad Name",       width="medium"),
                "CLIENT":          st.column_config.TextColumn("Operator",       width="medium"),
                "COUNTY":          st.column_config.TextColumn("County",         width="small"),
                "n_waste_records": st.column_config.NumberColumn("Waste Recs",   format="%d",   width="small"),
                "total_tons":      st.column_config.NumberColumn("Tons",         format="%.1f", width="small"),
                "total_bbls":      st.column_config.NumberColumn("Bbls",         format="%.1f", width="small"),
                "n_candidates":    st.column_config.NumberColumn("Candidates",   format="%d",   width="small"),
                "n_approved":      st.column_config.NumberColumn("Approved",     format="%d",   width="small"),
                "n_pending":       st.column_config.NumberColumn("Pending",      format="%d",   width="small"),
            },
            key="pad_editor",
        )

        # Auto-save: detect any change to the included column and persist immediately.
        # A separate save button doesn't work reliably because the button-click rerun
        # re-initialises the data_editor from disk before the save executes.
        old_included_arr = cp["included"].values
        new_included_arr = edited["included"].values
        if not (old_included_arr == new_included_arr).all():
            old_inc_set = set(cp.loc[cp["included"], "pad_WELL_PAD_ID"])
            cp["included"] = new_included_arr
            new_inc_set = set(cp.loc[cp["included"], "pad_WELL_PAD_ID"])
            save_cpads(sp, cp)
            newly_checked = new_inc_set - old_inc_set
            if newly_checked:
                n = sync_auto_links(sp, matched_links, pad_ids=newly_checked)
                st.toast(f"Saved · added {n} auto-link(s) for {len(newly_checked)} pad(s).")
            else:
                st.toast("Pad list saved.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Candidate Links
# ══════════════════════════════════════════════════════════════════════════════
with tab_links:
    if not sp:
        st.info("Load or create a session in the **Session** tab first.")
    else:
        st.subheader(f"Candidate Links — {session_meta.get('facility_name', '')}")
        cl_raw = load_clinks(sp)
        cp_cl_tab = load_cpads(sp)
        # Normalise to str to avoid int/float type mismatches after CSV round-trip
        included_ids_str = set(
            str(v) for v in cp_cl_tab.loc[cp_cl_tab["included"], "pad_WELL_PAD_ID"]
        )
        cl = cl_raw[cl_raw["pad_WELL_PAD_ID"].astype(str).isin(included_ids_str)].copy()

        with st.expander("🔍 Debug info"):
            st.write(f"**candidate_links.csv rows (total):** {len(cl_raw)}")
            st.write(f"**included pad IDs ({len(included_ids_str)}):** {sorted(included_ids_str)[:10]}")
            if not cl_raw.empty:
                sample_ids = sorted(cl_raw["pad_WELL_PAD_ID"].astype(str).unique())[:10]
                st.write(f"**pad_WELL_PAD_ID values in links CSV:** {sample_ids}")
                st.write(f"**dtype in links CSV:** {cl_raw['pad_WELL_PAD_ID'].dtype}")
            st.write(f"**dtype in pads CSV:** {cp_cl_tab['pad_WELL_PAD_ID'].dtype}")
            st.write(f"**rows after filter:** {len(cl)}")
            st.write("**Raw included values (first 5):**")
            raw_cp = pd.read_csv(sp / "candidate_pads.csv")
            st.dataframe(raw_cp[["pad_WELL_PAD_ID", "included"]].head())

        if not included_ids_str:
            st.info("No included pads yet. Check the **Include** box for one or more pads in the **Candidate Pads** tab and save.")
        elif cl.empty:
            st.info(
                f"{len(included_ids_str)} pad(s) included but no auto-links found for them. "
                "Use **Add manual links** below to assign lab reports manually."
            )
        else:
            n_p = (cl["status"] == "pending").sum()
            n_a = (cl["status"] == "approved").sum()
            n_r = (cl["status"] == "rejected").sum()
            st.caption(
                f"⏳ {n_p} pending  ·  ✅ {n_a} approved  ·  "
                f"❌ {n_r} rejected  ·  {len(cl)} for {len(included_ids_str)} included pad(s)"
            )

        sf_col, ba_col, br_col = st.columns([4, 1, 1])
        with sf_col:
            status_filter = st.multiselect(
                "Show status",
                ["pending", "approved", "rejected"],
                default=["pending", "approved", "rejected"],
                key="cl_filter",
            )
        with ba_col:
            if st.button("✅ Approve All", key="cl_approve_all"):
                cl_upd = load_clinks(sp)
                mask = cl_upd["pad_WELL_PAD_ID"].astype(str).isin(included_ids_str)
                cl_upd.loc[mask, "status"] = "approved"
                save_clinks(sp, cl_upd)
                st.rerun()
        with br_col:
            if st.button("❌ Reject All", key="cl_reject_all"):
                cl_upd = load_clinks(sp)
                mask = cl_upd["pad_WELL_PAD_ID"].astype(str).isin(included_ids_str)
                cl_upd.loc[mask, "status"] = "rejected"
                save_clinks(sp, cl_upd)
                st.rerun()

        disp = cl[cl["status"].isin(status_filter)].copy() if status_filter else cl.copy()
        disp[""] = disp["status"].map(status_icon)
        disp["pdf"] = disp.apply(pdf_url, axis=1)
        disp = disp.reset_index(drop=True)

        ev_cl = st.dataframe(
            disp[[
                "", "pdf", "original_filename", "lab_report_id",
                "pad_WELL_PAD_ID", "project_name", "lab_name", "client_name",
                "confidence", "loc_score", "source", "notes",
            ]],
            use_container_width=True,
            height=320,
            on_select="rerun",
            selection_mode="single-row",
            key="cl_table",
            column_config={
                "":               st.column_config.TextColumn("",           width="small"),
                "pdf":            st.column_config.LinkColumn("PDF",        display_text="Open", width="small"),
                "original_filename": st.column_config.TextColumn("Filename", width="large"),
                "lab_report_id":  st.column_config.TextColumn("Report ID", width="small"),
                "pad_WELL_PAD_ID": st.column_config.NumberColumn("Pad ID", format="%d",   width="small"),
                "project_name":   st.column_config.TextColumn("Project",   width="medium"),
                "lab_name":       st.column_config.TextColumn("Lab",       width="medium"),
                "client_name":    st.column_config.TextColumn("Client",    width="medium"),
                "confidence":     st.column_config.TextColumn("Conf",      width="small"),
                "loc_score":      st.column_config.NumberColumn("Score",   format="%.1f", width="small"),
                "source":         st.column_config.TextColumn("Source",    width="small"),
                "notes":          st.column_config.TextColumn("Notes",     width="medium"),
            },
        )

        sel_cl = ev_cl.selection.rows if ev_cl else []
        if sel_cl and sel_cl[0] < len(disp):
            row = disp.iloc[sel_cl[0]]
            st.divider()

            cp_cl = load_cpads(sp)
            pad_name_map_cl = dict(zip(cp_cl["pad_WELL_PAD_ID"], cp_cl["pad_WELL_PAD"]))
            pad_label = pad_name_map_cl.get(row["pad_WELL_PAD_ID"], f"Pad ID {row['pad_WELL_PAD_ID']}")

            try:
                pg = int(row["first_page"]) if pd.notna(row.get("first_page")) else 1
            except (ValueError, TypeError):
                pg = 1

            st.caption(
                f"{row[''] or ''} **{row['original_filename']}**"
                f"  ·  Report ID: {row.get('lab_report_id') or '—'}"
                f"  ·  Pad: {pad_label}"
                f"  ·  Project: {row.get('project_name') or '—'}"
                f"  ·  Lab: {row.get('lab_name') or '—'}"
                f"  ·  Client: {row.get('client_name') or '—'}"
                f"  ·  Conf: {row.get('confidence') or '—'} ({row.get('loc_score') or '—'})"
                f"  ·  [Open PDF p.{pg}]({row['pdf']})"
            )

            notes_key_cl = f"cl_notes_{row['original_filename']}_{row.get('lab_report_id', '')}_{row['pad_WELL_PAD_ID']}"
            act1, act2, act3, act4 = st.columns([2, 1, 1, 1])
            with act1:
                notes_val_cl = st.text_input("Notes", value=str(row.get("notes") or ""), key=notes_key_cl, label_visibility="collapsed", placeholder="Notes (optional)")

            def _set_cl_status(new_status: str):
                cl_upd = load_clinks(sp)
                mask = (
                    (cl_upd["original_filename"] == row["original_filename"])
                    & (cl_upd["lab_report_id"].astype(str) == str(row.get("lab_report_id") or ""))
                    & (cl_upd["pad_WELL_PAD_ID"] == row["pad_WELL_PAD_ID"])
                )
                cl_upd.loc[mask, "status"] = new_status
                cl_upd.loc[mask, "notes"] = notes_val_cl
                save_clinks(sp, cl_upd)

            with act2:
                if st.button("✅ Approve", key="cl_approve", type="primary"):
                    _set_cl_status("approved")
                    st.rerun()
            with act3:
                if st.button("❌ Reject", key="cl_reject"):
                    _set_cl_status("rejected")
                    st.rerun()
            with act4:
                if st.button("⏳ Pending", key="cl_pending"):
                    _set_cl_status("pending")
                    st.rerun()

            with st.spinner("Loading analyte data…"):
                lab_df = _load_lab_results()

            sel_fn  = row["original_filename"]
            sel_rid = str(row.get("lab_report_id") or "")
            analytes = lab_df[
                (lab_df["original_filename"] == sel_fn)
                & (lab_df["lab_report_id"].astype(str) == sel_rid)
            ].sort_values(["original_page", "lab_sample_id"]).reset_index(drop=True)

            if analytes.empty:
                st.warning("No analyte rows found for this (filename, report ID) pair.")
            else:
                st.caption(f"{len(analytes):,} analyte rows  ·  pages {analytes['original_page'].min()}–{analytes['original_page'].max()}")
                st.dataframe(
                    analytes[[
                        "lab_sample_id", "analyte_norm",
                        "result_value", "result_flag",
                        "project_name", "client_name", "received_date",
                    ]],
                    use_container_width=True,
                    height=400,
                    column_config={
                        "lab_sample_id":  st.column_config.TextColumn("Sample ID",    width="medium"),
                        "analyte_norm":   st.column_config.TextColumn("Analyte",      width="medium"),
                        "result_value":   st.column_config.NumberColumn("Result",     format="%.4g", width="small"),
                        "result_flag":    st.column_config.TextColumn("Flag",         width="small"),
                        "project_name":   st.column_config.TextColumn("Project",      width="medium"),
                        "client_name":    st.column_config.TextColumn("Client",       width="medium"),
                        "received_date":  st.column_config.TextColumn("Received",     width="small"),
                    },
                )
        else:
            st.caption("Select a row to see its analyte data.")

        st.divider()

        # ── Add manual links for any pad ──────────────────────────────────────
        with st.expander("Add manual links"):
            cp = load_cpads(sp)
            included_pads = cp[cp["included"]].copy()

            # Show unlinked pads first, then the rest
            all_candidate_ids = set(cl["pad_WELL_PAD_ID"])
            included_pads["_has_link"] = included_pads["pad_WELL_PAD_ID"].isin(all_candidate_ids)
            included_pads = included_pads.sort_values("_has_link").reset_index(drop=True)

            pad_options = {
                f"{'✓' if row['_has_link'] else '○'} {row['pad_WELL_PAD']} (ID {int(row['pad_WELL_PAD_ID'])})": row["pad_WELL_PAD_ID"]
                for _, row in included_pads.iterrows()
            }
            if not pad_options:
                st.info("No included pads. Adjust the pad list in Candidate Pads.")
            else:
                st.caption("○ = no candidates yet  ·  ✓ = has candidates")
                selected_label = st.selectbox(
                    "Target pad", list(pad_options.keys()), key="manual_pad_sel"
                )
                selected_pad_id = pad_options[selected_label]
                selected_pad_name = included_pads.loc[
                    included_pads["pad_WELL_PAD_ID"] == selected_pad_id, "pad_WELL_PAD"
                ].iloc[0]

                # Ranked suggestions: no_match items whose best-guess pad name
                # shares tokens with the selected pad
                nm = no_match_labs.copy()
                pad_tokens = set(re.findall(r"\w+", str(selected_pad_name).lower())) - {"the", "and", "or", "of", "at"}
                if "matched_pad_name" in nm.columns:
                    nm["_overlap"] = nm["matched_pad_name"].apply(
                        lambda v: len(pad_tokens & set(re.findall(r"\w+", str(v).lower()))) if pd.notna(v) else 0
                    )
                    ranked = nm[nm["_overlap"] > 0].sort_values(["_overlap", "loc_score"], ascending=False)
                else:
                    ranked = pd.DataFrame()

                search_q = st.text_input(
                    "Search no-match lab units (filename, project, client)",
                    placeholder="e.g. Kingsley, Chief, 26R-100",
                    key="nm_search",
                )

                if search_q.strip():
                    q = search_q.strip()
                    mask = (
                        nm["original_filename"].str.contains(q, case=False, na=False)
                        | nm.get("project_name", pd.Series(dtype=str)).str.contains(q, case=False, na=False)
                        | nm.get("client_name", pd.Series(dtype=str)).str.contains(q, case=False, na=False)
                    )
                    show_nm = nm[mask].sort_values("loc_score", ascending=False)
                    st.write(f"**Search results ({len(show_nm)}):**")
                elif not ranked.empty:
                    show_nm = ranked
                    st.write(f"**Ranked suggestions ({len(ranked)}) — name token overlap:**")
                else:
                    show_nm = nm.sort_values("loc_score", ascending=False)
                    st.write("**All no-match units (sorted by score):**")

                show_nm = show_nm.head(150).reset_index(drop=True)
                show_nm["pdf"] = show_nm.apply(pdf_url, axis=1)

                ev_nm = st.dataframe(
                    show_nm[[
                        "pdf", "original_filename", "lab_report_id",
                        "project_name", "lab_name", "client_name",
                        "matched_pad_name", "loc_score",
                    ]],
                    use_container_width=True,
                    height=300,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="nm_table",
                    column_config={
                        "pdf":               st.column_config.LinkColumn("PDF",        display_text="Open", width="small"),
                        "original_filename": st.column_config.TextColumn("Filename",   width="large"),
                        "lab_report_id":     st.column_config.TextColumn("Report ID",  width="small"),
                        "project_name":      st.column_config.TextColumn("Project",    width="medium"),
                        "lab_name":          st.column_config.TextColumn("Lab",        width="medium"),
                        "client_name":       st.column_config.TextColumn("Client",     width="medium"),
                        "matched_pad_name":  st.column_config.TextColumn("Best Guess", width="medium"),
                        "loc_score":         st.column_config.NumberColumn("Score",    format="%.1f", width="small"),
                    },
                )

                sel_nm = ev_nm.selection.rows if ev_nm else []
                if sel_nm:
                    nm_row = show_nm.iloc[sel_nm[0]]
                    st.info(
                        f"Selected: **{nm_row['original_filename']}**  "
                        f"(Report ID: {nm_row.get('lab_report_id') or '—'})  →  "
                        f"pad **{selected_pad_name}**"
                    )
                    if st.button("Add to candidate links", type="primary", key="btn_add_manual"):
                        cl_now = load_clinks(sp)
                        key_fn  = nm_row["original_filename"]
                        key_rid = str(nm_row.get("lab_report_id") or "")
                        already = (
                            (cl_now["original_filename"] == key_fn)
                            & (cl_now["lab_report_id"].astype(str) == key_rid)
                            & (cl_now["pad_WELL_PAD_ID"] == selected_pad_id)
                        ).any()
                        if already:
                            st.warning("This (filename, report_id, pad) combination is already in the list.")
                        else:
                            new_row = {c: nm_row.get(c) for c in LINK_COLS}
                            new_row.update(
                                pad_WELL_PAD_ID=selected_pad_id,
                                source="manual",
                                status="pending",
                                notes="",
                            )
                            cl_now = pd.concat(
                                [cl_now, pd.DataFrame([new_row], columns=LINK_COLS)],
                                ignore_index=True,
                            )
                            save_clinks(sp, cl_now)
                            st.success("Added to candidate links.")
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Review
# ══════════════════════════════════════════════════════════════════════════════
with tab_review:
    if not sp:
        st.info("Load or create a session in the **Session** tab first.")
    else:
        st.subheader(f"Review — {session_meta.get('facility_name', '')}")
        cl = load_clinks(sp)
        cp = load_cpads(sp)
        pad_name_map = dict(zip(cp["pad_WELL_PAD_ID"], cp["pad_WELL_PAD"]))

        n_p = (cl["status"] == "pending").sum()
        n_a = (cl["status"] == "approved").sum()
        n_r = (cl["status"] == "rejected").sum()
        st.caption(f"⏳ {n_p} pending  ·  ✅ {n_a} approved  ·  ❌ {n_r} rejected")

        rc1, rc2 = st.columns([3, 1])
        with rc1:
            rev_status = st.multiselect(
                "Show",
                ["pending", "approved", "rejected"],
                default=["pending"],
                key="rev_filter",
            )
        with rc2:
            rev_sort = st.selectbox(
                "Sort",
                ["score ↓", "pad", "confidence"],
                key="rev_sort",
            )

        q = cl[cl["status"].isin(rev_status)].copy() if rev_status else cl.copy()
        if rev_sort == "score ↓":
            q = q.sort_values("loc_score", ascending=False)
        elif rev_sort == "pad":
            q = q.sort_values("pad_WELL_PAD_ID")
        elif rev_sort == "confidence":
            q["_co"] = q["confidence"].map({"high": 0, "medium": 1, "low": 2}).fillna(3)
            q = q.sort_values(["_co", "loc_score"], ascending=[True, False]).drop(columns="_co")
        q = q.reset_index(drop=True)

        if q.empty:
            st.info("No items match the current filter.")
        else:
            q[""] = q["status"].map(status_icon)
            q["pdf"] = q.apply(pdf_url, axis=1)
            q["pad_name"] = q["pad_WELL_PAD_ID"].map(pad_name_map)

            ev_rev = st.dataframe(
                q[[
                    "", "pdf", "pad_name", "original_filename",
                    "lab_report_id", "project_name", "lab_name", "client_name",
                    "confidence", "loc_score", "source",
                ]],
                use_container_width=True,
                height=280,
                on_select="rerun",
                selection_mode="single-row",
                key="rev_table",
                column_config={
                    "":               st.column_config.TextColumn("",          width="small"),
                    "pdf":            st.column_config.LinkColumn("PDF",       display_text="Open", width="small"),
                    "pad_name":       st.column_config.TextColumn("Pad",       width="medium"),
                    "original_filename": st.column_config.TextColumn("Filename", width="large"),
                    "lab_report_id":  st.column_config.TextColumn("Report ID", width="small"),
                    "project_name":   st.column_config.TextColumn("Project",   width="medium"),
                    "lab_name":       st.column_config.TextColumn("Lab",       width="medium"),
                    "client_name":    st.column_config.TextColumn("Client",    width="medium"),
                    "confidence":     st.column_config.TextColumn("Conf",      width="small"),
                    "loc_score":      st.column_config.NumberColumn("Score",   format="%.1f", width="small"),
                    "source":         st.column_config.TextColumn("Source",    width="small"),
                },
            )

            sel_rev = ev_rev.selection.rows if ev_rev else []
            if not sel_rev or sel_rev[0] >= len(q):
                st.info("Select a row above to review it.")
            else:
                row = q.iloc[sel_rev[0]]
                pad_id = row["pad_WELL_PAD_ID"]

                st.divider()
                left, right = st.columns(2)

                with left:
                    st.subheader("Lab report")
                    st.write(f"**Filename:** {row['original_filename']}")
                    st.write(f"**Report ID:** {row.get('lab_report_id') or '—'}")
                    st.write(f"**Project:** {row.get('project_name') or '—'}")
                    st.write(f"**Lab:** {row.get('lab_name') or '—'}")
                    st.write(f"**Client:** {row.get('client_name') or '—'}")
                    st.write(
                        f"**Confidence:** {row.get('confidence') or '—'}"
                        f"  ·  Score: {row.get('loc_score') or '—'}"
                        f"  ·  Source: {row.get('source') or '—'}"
                    )
                    try:
                        pg = int(row["first_page"]) if pd.notna(row.get("first_page")) else 1
                    except (ValueError, TypeError):
                        pg = 1
                    st.markdown(f"**[Open PDF — page {pg}]({row['pdf']})**")
                    st.write(f"**Pad:** {row.get('pad_name') or pad_id}")

                with right:
                    st.subheader("Pad waste records at facility")
                    fac_query = session_meta.get("facility_query", "")
                    pad_waste = waste[
                        (waste["pad_WELL_PAD_ID"] == pad_id)
                        & waste["WASTE_FACILITY_NAME"].str.contains(fac_query, case=False, na=False)
                    ].drop(columns="pad_WELL_PAD_ID").reset_index(drop=True)
                    if pad_waste.empty:
                        st.info("No waste records for this pad at the facility.")
                    else:
                        st.dataframe(
                            pad_waste,
                            use_container_width=True,
                            height=200,
                            column_config={
                                "PERMIT_NUM":      st.column_config.TextColumn("Well API",   width="medium"),
                                "WELL_NO":         st.column_config.TextColumn("Well",       width="small"),
                                "_period_label":   st.column_config.TextColumn("Period",     width="small"),
                                "PRODUCT_TYPE":    st.column_config.TextColumn("Waste Type", width="medium"),
                                "QUANTITY":        st.column_config.NumberColumn("Qty",      format="%.1f", width="small"),
                                "UNITS":           st.column_config.TextColumn("Units",      width="small"),
                                "DISPOSAL_METHOD": st.column_config.TextColumn("Disposal",   width="medium"),
                                "WASTE_FACILITY_NAME": st.column_config.TextColumn("Facility", width="large"),
                            },
                        )

                st.divider()
                notes_key = f"notes_{row['original_filename']}_{row.get('lab_report_id', '')}_{pad_id}"
                notes_val = st.text_input(
                    "Notes (optional)",
                    value=str(row.get("notes") or ""),
                    key=notes_key,
                )

                def _update_status(new_status: str):
                    cl_upd = load_clinks(sp)
                    mask = (
                        (cl_upd["original_filename"] == row["original_filename"])
                        & (cl_upd["lab_report_id"].astype(str) == str(row.get("lab_report_id") or ""))
                        & (cl_upd["pad_WELL_PAD_ID"] == pad_id)
                    )
                    cl_upd.loc[mask, "status"] = new_status
                    cl_upd.loc[mask, "notes"] = notes_val
                    save_clinks(sp, cl_upd)

                btn_a, btn_r = st.columns(2)
                with btn_a:
                    if st.button("✅ Approve", type="primary", key="btn_approve"):
                        _update_status("approved")
                        st.rerun()
                with btn_r:
                    if st.button("❌ Reject", key="btn_reject"):
                        _update_status("rejected")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Output
# ══════════════════════════════════════════════════════════════════════════════
with tab_output:
    if not sp:
        st.info("Load or create a session in the **Session** tab first.")
    else:
        st.subheader(f"Output — {session_meta.get('facility_name', '')}")
        cl = load_clinks(sp)

        approved = cl[cl["status"] == "approved"].copy()
        n_pending  = (cl["status"] == "pending").sum()
        n_rejected = (cl["status"] == "rejected").sum()

        col_a, col_p, col_r = st.columns(3)
        col_a.metric("Approved", len(approved))
        col_p.metric("Pending",  n_pending)
        col_r.metric("Rejected", n_rejected)

        if n_pending:
            st.warning(f"{n_pending} candidates are still pending review.")

        out_dir        = sp / "output"
        out_links_csv  = out_dir / "approved_links.csv"
        out_results_pq = out_dir / "lab_results.parquet"

        if approved.empty:
            st.info("No approved links yet. Use the **Review** tab to approve candidates.")
        else:
            if st.button("Compile output", type="primary", key="btn_compile"):
                with st.spinner("Loading lab results parquet…"):
                    lab_df = _load_lab_results()

                approved_fn  = approved["original_filename"].astype(str)
                approved_rid = approved["lab_report_id"].astype(str)
                approved_keys = set(zip(approved_fn, approved_rid))

                # Build dict: (fn, rid) → pad_WELL_PAD_ID (first match if duplicates)
                pad_lu = {
                    (str(fn), str(rid)): pid
                    for fn, rid, pid in zip(
                        approved["original_filename"],
                        approved["lab_report_id"],
                        approved["pad_WELL_PAD_ID"],
                    )
                }

                lab_fn  = lab_df["original_filename"].astype(str)
                lab_rid = lab_df["lab_report_id"].astype(str)
                mask = pd.Series(
                    [(fn, rid) in approved_keys for fn, rid in zip(lab_fn, lab_rid)],
                    index=lab_df.index,
                )
                filtered = lab_df[mask].copy()
                filtered["pad_WELL_PAD_ID"] = [
                    pad_lu.get((str(fn), str(rid)))
                    for fn, rid in zip(
                        filtered["original_filename"].astype(str),
                        filtered["lab_report_id"].astype(str),
                    )
                ]

                approved.to_csv(out_links_csv, index=False)
                filtered.to_parquet(out_results_pq, index=False)

                st.success(
                    f"Compiled: **{len(approved)}** approved links  ·  "
                    f"**{len(filtered):,}** lab result rows"
                )
                st.write(f"Output folder: `{out_dir}`")

            # Show existing output
            if out_links_csv.exists():
                st.divider()
                links_out = pd.read_csv(out_links_csv)
                st.write(
                    f"**Last compiled output:**  "
                    f"`approved_links.csv` — {len(links_out):,} rows  ·  "
                    f"`lab_results.parquet`"
                )
                with open(out_links_csv, "rb") as f:
                    st.download_button(
                        "Download approved_links.csv",
                        data=f,
                        file_name=f"{sp.name}_approved_links.csv",
                        mime="text/csv",
                        key="dl_links",
                    )

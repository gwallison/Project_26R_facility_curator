"""
Assign synthetic lab_report_id values to lab result rows that are missing them.

Uses metadata clustering (lab_name, client_name, received_date) to identify
distinct report boundaries within each file. Files where clustering is
ambiguous are flagged for manual review.

Outputs (in data/):
  report_id_assignments.csv  — per-sample mapping: (filename, sample) → rid
  report_id_page_map.csv     — per-page mapping: (filename, page) → rid
  ambiguous_report_files.csv — files needing manual review

Usage:
    python assign_report_ids.py
"""

import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

LAB_PARQUET = Path(
    r"G:\My Drive\sandbox\26R\data_cleanup_26R\processed"
    r"\lab_results_result_parsed.parquet"
)


def _cluster_file(rows: pd.DataFrame) -> list[dict]:
    """Cluster empty-rid rows in a single file by metadata signature.

    Returns a list of cluster dicts with keys:
        assigned_rid, method, samples, min_page, max_page, meta_sig
    """
    samples = rows.groupby("lab_sample_id").agg(
        lab_name=("lab_name", "first"),
        client_name=("client_name", "first"),
        received_date=("received_date", "first"),
        min_page=("original_page", "min"),
        max_page=("original_page", "max"),
    ).reset_index()

    samples["_meta"] = (
        samples["lab_name"].fillna("?") + "|"
        + samples["client_name"].fillna("?") + "|"
        + samples["received_date"].fillna("?").astype(str)
    )

    n_clusters = samples["_meta"].nunique()

    if n_clusters == 1:
        all_unknown = samples["_meta"].iloc[0] == "?|?|?"
        if all_unknown and len(samples) > 3:
            return []  # ambiguous
        first_sid = sorted(samples["lab_sample_id"])[0]
        return [{
            "assigned_rid": f"s:{first_sid}",
            "method": "single_report",
            "samples": list(samples["lab_sample_id"]),
            "min_page": int(samples["min_page"].min()),
            "max_page": int(samples["max_page"].max()),
            "meta_sig": samples["_meta"].iloc[0],
        }]

    # Multiple metadata signatures — check for page overlap
    cluster_ranges = (
        samples.groupby("_meta")
        .agg(min_page=("min_page", "min"), max_page=("max_page", "max"))
        .sort_values("min_page")
    )
    ranges = list(zip(cluster_ranges["min_page"], cluster_ranges["max_page"]))
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            return []  # overlapping — ambiguous

    clusters = []
    for meta_sig, grp in samples.groupby("_meta"):
        first_sid = sorted(grp["lab_sample_id"])[0]
        clusters.append({
            "assigned_rid": f"s:{first_sid}",
            "method": "metadata_cluster",
            "samples": list(grp["lab_sample_id"]),
            "min_page": int(grp["min_page"].min()),
            "max_page": int(grp["max_page"].max()),
            "meta_sig": meta_sig,
        })
    return clusters


def main():
    print("Loading lab results…")
    lab = pd.read_parquet(LAB_PARQUET)
    lab["_rid"] = lab["lab_report_id"].fillna("").astype(str).str.strip()

    empty_files = sorted(lab[lab["_rid"] == ""]["original_filename"].unique())
    print(f"Files with empty lab_report_id: {len(empty_files)}")

    sample_rows = []  # (filename, lab_sample_id, assigned_rid, method)
    page_rows = []    # (filename, page, assigned_rid)
    ambiguous = []    # (filename, reason, n_samples)

    for fn in empty_files:
        rows = lab[(lab["original_filename"] == fn) & (lab["_rid"] == "")]
        clusters = _cluster_file(rows)

        if not clusters:
            n_samples = rows["lab_sample_id"].nunique()
            meta_sigs = (
                rows.groupby("lab_sample_id")
                .agg(lab_name=("lab_name", "first"),
                     client_name=("client_name", "first"),
                     received_date=("received_date", "first"))
            )
            all_unknown = (
                meta_sigs["lab_name"].isna().all()
                and meta_sigs["client_name"].isna().all()
                and meta_sigs["received_date"].isna().all()
            )
            reason = "all_unknown_metadata" if all_unknown else "overlapping_page_ranges"
            ambiguous.append({
                "original_filename": fn,
                "reason": reason,
                "n_samples": n_samples,
            })
            continue

        for cl in clusters:
            for sid in cl["samples"]:
                sample_rows.append({
                    "original_filename": fn,
                    "lab_sample_id": sid,
                    "assigned_rid": cl["assigned_rid"],
                    "method": cl["method"],
                })
            for pg in range(cl["min_page"], cl["max_page"] + 1):
                page_rows.append({
                    "original_filename": fn,
                    "original_page": pg,
                    "assigned_rid": cl["assigned_rid"],
                })

    # ── Phase 2: resolve link rows where the file has real report IDs but ──
    # ── the linker didn't propagate them (page-range lookup)            ──
    links = pd.read_parquet(
        Path(r"G:\My Drive\sandbox\26R\link_DEP_waste_to_labs\data\links.parquet")
    )
    link_rid = links["lab_report_id"].fillna("").astype(str).str.strip()
    empty_link_fns = set(links.loc[link_rid == "", "original_filename"])
    assigned_fns = {r["original_filename"] for r in sample_rows}
    ambig_fns = {a["original_filename"] for a in ambiguous}
    orphan_fns = empty_link_fns - assigned_fns - ambig_fns

    if orphan_fns:
        nonempty_lab = lab[lab["_rid"] != ""]
        ranges = nonempty_lab.groupby(["original_filename", "_rid"]).agg(
            min_page=("original_page", "min"),
            max_page=("original_page", "max"),
        ).reset_index()

        orphan_resolved = 0
        for _, lr in links[
            (link_rid == "") & links["original_filename"].isin(orphan_fns)
        ].iterrows():
            fn = lr["original_filename"]
            fp = int(lr["first_page"])
            candidates = ranges[
                (ranges["original_filename"] == fn)
                & (ranges["min_page"] <= fp)
                & (ranges["max_page"] >= fp)
            ]
            if len(candidates) == 1:
                real_rid = candidates.iloc[0]["_rid"]
                page_rows.append({
                    "original_filename": fn,
                    "original_page": fp,
                    "assigned_rid": real_rid,
                })
                orphan_resolved += 1
        print(f"Phase 2: resolved {orphan_resolved} link rows via page-range lookup")

    # Write outputs
    sample_df = pd.DataFrame(sample_rows)
    page_df = pd.DataFrame(page_rows).drop_duplicates()
    ambig_df = pd.DataFrame(ambiguous)

    sample_df.to_csv(DATA_DIR / "report_id_assignments.csv", index=False)
    page_df.to_csv(DATA_DIR / "report_id_page_map.csv", index=False)
    ambig_df.to_csv(DATA_DIR / "ambiguous_report_files.csv", index=False)

    n_assigned = sample_df["original_filename"].nunique()
    n_samples = len(sample_df)
    n_ambig = len(ambig_df)
    print(f"\nAssigned synthetic IDs for {n_assigned} files ({n_samples} samples)")
    print(f"Ambiguous files flagged for review: {n_ambig}")
    print(f"\nOutputs:")
    print(f"  {DATA_DIR / 'report_id_assignments.csv'}")
    print(f"  {DATA_DIR / 'report_id_page_map.csv'}")
    print(f"  {DATA_DIR / 'ambiguous_report_files.csv'}")


if __name__ == "__main__":
    main()

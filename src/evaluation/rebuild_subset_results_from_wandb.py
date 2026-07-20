"""
Rebuild the subset-selection results CSV from W&B run history.

The local summary CSV of subset-selection pretraining runs was lost, but every
run (config, per-epoch history, summary, and captured stdout) is still on
W&B at boulder-rolf-lab/GeoYFCC-Text-Subset. This script re-derives one row
per run from:

  - run.config          -> subset_size (B), num_domains (K), domain_selection_method,
                            val_subset_size (V), learning_rate, batch_size, num_epochs
  - run.name            -> seed, tgt_domain, and OT params (embedding_type, method,
                            reg, iter, metric, norm), which aren't in config
  - run.summary/history -> best_epoch and the val/train accuracy (+top3/top5) at
                            that epoch, i.e. the checkpointed model's performance
  - output.log          -> candidate domains, matched country names, and the
                            train/val mask sizes actually used, printed by
                            subset_selection.py / pretrain_by_domain_subset.py

Usage:
    python src/evaluation/rebuild_subset_results_from_wandb.py \
        --entity boulder-rolf-lab --project GeoYFCC-Text-Subset \
        --out results/subset/subset_selection_results_from_wandb.csv
"""

import argparse
import csv
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import wandb

NAME_PATTERNS = {
    "seed": re.compile(r"_seed(\d+)$"),
    "tgt_domain": re.compile(r"_tgt(\d+)"),
    "ot": re.compile(
        r"_OT_(?P<emb>.+?)_(?P<method>sinkhorn)_(?P<reg>[0-9.]+)_(?P<iter>\d+)"
        r"_(?P<metric>cosine)_(?P<norm>[a-zA-Z_]+?)(?=_V\d|_tgt\d|_seed\d|$)"
    ),
}

LOG_PATTERNS = {
    "domains": re.compile(r"\[DOMAINS\]\s*K=(\d+)\s*->\s*\[(.*?)\]"),
    "country": re.compile(r"Country names matched:\s*(\[.*?\])"),
    "train_mask": re.compile(r"\[MASK\]\s*Train selected=(\d+)\s*/\s*budget=(\d+)"),
    "val_mask": re.compile(
        r"\[MASK\]\s*Val selected=(\d+)(?:\s*/\s*budget=(\d+)|\s*\(union full\))"
    ),
}

ALL_COLUMNS = [
    "run_id", "run_name", "run_url", "state", "created_at",
    "time_taken", "epochs_trained",
    "dataset", "domain_type", "model",
    "subset_size_B", "num_domains_K", "domain_selection_method", "val_subset_size_V",
    "tgt_domain", "seed",
    "ot_embedding_type", "ot_method", "ot_reg", "ot_iter", "ot_metric", "ot_norm",
    "candidate_domain_ids", "candidate_domain_countries",
    "train_selected", "train_budget", "val_selected", "val_budget",
    "best_epoch", "best_val_acc", "best_val_top3_acc", "best_val_top5_acc",
    "best_train_acc", "best_train_top3_acc", "best_train_top5_acc",
    "final_epoch", "final_val_acc", "final_train_acc",
]

# Run-bookkeeping fields kept internally (needed to identify/dedupe retried runs)
# but stripped from the written CSVs since they're not analysis-relevant.
_DROP_FROM_OUTPUT = {"run_id", "run_name", "run_url", "state", "created_at", "time_taken"}
OUTPUT_COLUMNS = [c for c in ALL_COLUMNS if c not in _DROP_FROM_OUTPUT]


def parse_name_fields(name):
    fields = {
        "seed": None, "tgt_domain": None,
        "ot_embedding_type": None, "ot_method": None, "ot_reg": None,
        "ot_iter": None, "ot_metric": None, "ot_norm": None,
    }
    m = NAME_PATTERNS["seed"].search(name)
    if m:
        fields["seed"] = m.group(1)
    m = NAME_PATTERNS["tgt_domain"].search(name)
    if m:
        fields["tgt_domain"] = m.group(1)
    m = NAME_PATTERNS["ot"].search(name)
    if m:
        fields["ot_embedding_type"] = m.group("emb")
        fields["ot_method"] = m.group("method")
        fields["ot_reg"] = m.group("reg")
        fields["ot_iter"] = m.group("iter")
        fields["ot_metric"] = m.group("metric")
        fields["ot_norm"] = m.group("norm")
    return fields


def parse_log_fields(log_text):
    fields = {
        "candidate_domain_ids": None, "candidate_domain_countries": None,
        "train_selected": None, "train_budget": None,
        "val_selected": None, "val_budget": None,
    }
    m = LOG_PATTERNS["domains"].search(log_text)
    if m:
        ids = re.findall(r"(?:np\.int64\()?(\d+)\)?", m.group(2))
        fields["candidate_domain_ids"] = "+".join(ids)

    countries = LOG_PATTERNS["country"].findall(log_text)
    if countries:
        # Only keep the first block of matches (train-mask lookup); the val-mask
        # lookup that follows repeats the same candidate domains.
        n = len(re.findall(r"\d+", fields["candidate_domain_ids"] or "")) or len(countries)
        first_block = countries[:n]
        names = []
        for block in first_block:
            names.extend(re.findall(r"'([^']*)'", block))
        fields["candidate_domain_countries"] = "+".join(names)

    m = LOG_PATTERNS["train_mask"].search(log_text)
    if m:
        fields["train_selected"], fields["train_budget"] = m.group(1), m.group(2)

    m = LOG_PATTERNS["val_mask"].search(log_text)
    if m:
        fields["val_selected"] = m.group(1)
        fields["val_budget"] = m.group(2) if m.group(2) else "union_full"

    return fields


def get_output_log_text(run, tmpdir):
    for f in run.files():
        if f.name == "output.log":
            try:
                path = f.download(root=tmpdir, replace=True).name
                with open(path, "r", errors="replace") as fh:
                    return fh.read()
            except Exception:
                return ""
    return ""


def build_row(run, tmpdir):
    name = run.name
    group = run.group or ""
    group_parts = group.split("/")
    dataset = group_parts[0] if len(group_parts) > 0 else None
    domain_type = group_parts[1] if len(group_parts) > 1 else None
    model = group_parts[3] if len(group_parts) > 3 else run.config.get("model")

    cfg = run.config
    summary = dict(run.summary)
    name_fields = parse_name_fields(name)

    log_text = get_output_log_text(run, tmpdir)
    log_fields = parse_log_fields(log_text)

    best_epoch = summary.get("best_epoch")
    best_val_acc = summary.get("val_acc")  # summary['val_acc'] is the *best* val acc
    best_train_acc = best_val_top3 = best_val_top5 = best_train_top3 = best_train_top5 = None

    if best_epoch is not None:
        try:
            history = run.history()
            if not history.empty and "epoch" in history.columns:
                row = history.loc[history["epoch"] == best_epoch - 1]
                if not row.empty:
                    row = row.iloc[0]
                    best_train_acc = row.get("train/acc")
                    best_val_top3 = row.get("val/top3_acc")
                    best_val_top5 = row.get("val/top5_acc")
                    best_train_top3 = row.get("train/top3_acc")
                    best_train_top5 = row.get("train/top5_acc")
        except Exception:
            pass

    return {
        "run_id": run.id,
        "run_name": name,
        "run_url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "time_taken": summary.get("time_taken"),
        "epochs_trained": summary.get("epochs_trained"),
        "dataset": dataset,
        "domain_type": domain_type,
        "model": model,
        "subset_size_B": cfg.get("subset_size"),
        "num_domains_K": cfg.get("num_domains"),
        "domain_selection_method": cfg.get("domain_selection_method"),
        "val_subset_size_V": cfg.get("val_subset_size"),
        "tgt_domain": name_fields["tgt_domain"],
        "seed": name_fields["seed"],
        "ot_embedding_type": name_fields["ot_embedding_type"],
        "ot_method": name_fields["ot_method"],
        "ot_reg": name_fields["ot_reg"],
        "ot_iter": name_fields["ot_iter"],
        "ot_metric": name_fields["ot_metric"],
        "ot_norm": name_fields["ot_norm"],
        "candidate_domain_ids": log_fields["candidate_domain_ids"],
        "candidate_domain_countries": log_fields["candidate_domain_countries"],
        "train_selected": log_fields["train_selected"],
        "train_budget": log_fields["train_budget"],
        "val_selected": log_fields["val_selected"],
        "val_budget": log_fields["val_budget"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_top3_acc": best_val_top3,
        "best_val_top5_acc": best_val_top5,
        "best_train_acc": best_train_acc,
        "best_train_top3_acc": best_train_top3,
        "best_train_top5_acc": best_train_top5,
        "final_epoch": summary.get("epoch"),
        "final_val_acc": summary.get("val/acc"),
        "final_train_acc": summary.get("train/acc"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="boulder-rolf-lab")
    parser.add_argument("--project", default="GeoYFCC-Text-Subset")
    parser.add_argument("--out", default="results/subset/subset_selection_results_from_wandb.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N runs (for testing)")
    parser.add_argument("--workers", type=int, default=12, help="Parallel threads for fetching run data")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.limit:
        runs = runs[: args.limit]

    print(f"Found {len(runs)} runs in {args.entity}/{args.project}")

    rows = []
    done = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build_row, run, tmpdir): run for run in runs}
            for fut in as_completed(futures):
                run = futures[fut]
                done += 1
                try:
                    rows.append(fut.result())
                except Exception as e:
                    print(f"  [WARN] failed to process run {run.id} ({run.name}): {e}")
                if done % 25 == 0:
                    print(f"  processed {done}/{len(runs)}")

    rows.sort(key=lambda r: (r["run_name"] or "", r["created_at"] or ""))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

    # Several configs were re-submitted (retries after a crash/OOM, or a later
    # no-op run that found the checkpoint already existed and skipped straight
    # to "already completed"). Collapse those to one row per distinct config
    # (identified by run_name, which encodes B/K/method/V/tgt/seed), preferring
    # a row that actually has best_val_acc, then a 'finished' state, then the
    # most recent attempt.
    df = pd.DataFrame(rows)
    df["_has_data"] = df["best_val_acc"].notna()
    df["_is_finished"] = df["state"] == "finished"
    df = df.sort_values(
        ["run_name", "_has_data", "_is_finished", "created_at"],
        ascending=[True, False, False, False],
    )
    dedup = df.groupby("run_name", as_index=False).first()

    dedup_out = args.out.replace(".csv", "_deduped.csv")
    if dedup_out == args.out:
        dedup_out = args.out + ".deduped.csv"
    dedup[OUTPUT_COLUMNS].to_csv(dedup_out, index=False)

    n_missing = dedup["best_val_acc"].isna().sum()
    print(
        f"Wrote {len(dedup)} deduplicated rows (one per config) to {dedup_out}; "
        f"{n_missing} configs have no recoverable performance data on W&B "
        f"(training predates full W&B logging or every attempt crashed)."
    )
    if n_missing:
        missing_names = dedup.loc[dedup["best_val_acc"].isna(), "run_name"].tolist()
        print("  Configs with no recoverable performance:")
        for n in missing_names:
            print(f"    - {n}")


if __name__ == "__main__":
    main()

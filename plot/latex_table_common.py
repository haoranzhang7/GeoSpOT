"""Shared labels and LaTeX table builder for the plot/build_*_table.py scripts."""

from pathlib import Path

DATASET_LABELS = {
    "geoyfcc_text": "GeoYFCC-Text",
    "geoyfcc_image": "GeoYFCC-Image",
    "fmow": "FMoW",
    "geode": "GeoDE",
}

EMBEDDING_LABELS = {
    "bert": ("BERT", "Text"),
    "geoclip": ("GeoCLIP", "Location"),
    "satclip": ("SatCLIP", "Location"),
}

DISTANCE_LABELS = {"ot": "OT", "mmd": "MMD", "fid": "FID"}


def format_mean_std_cell(value):
    return "--" if value is None else f"${value[0]:.3f} \\pm {value[1]:.3f}$"


def to_latex(header, rows, caption, label):
    """rows: list of lists of pre-formatted cell strings, one row per output line."""
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\begin{{tabular}}{{{'l' * len(header)}}}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def build_dataset_embedding_table(datasets, embedding_types, column_keys, column_labels, value_fn, caption, label):
    """One row per (dataset, embedding_type), one column per column_key.
    value_fn(dataset, embedding_type, column_key) -> (mean, std) or None."""
    show_dataset_col = len(datasets) > 1
    header = (["Dataset"] if show_dataset_col else []) + ["Modality", "Embedding"] + \
        [column_labels[k] for k in column_keys]

    rows = []
    for dataset in datasets:
        for embedding_type in embedding_types:
            emb_name, modality = EMBEDDING_LABELS[embedding_type]
            cells = ([DATASET_LABELS.get(dataset, dataset)] if show_dataset_col else []) + [modality, emb_name]
            cells += [format_mean_std_cell(value_fn(dataset, embedding_type, k)) for k in column_keys]
            rows.append(cells)
    return to_latex(header, rows, caption, label)


def write_and_print(latex, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    print(latex)
    print(f"\nSaved to {out_path}")

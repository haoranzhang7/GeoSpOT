#!/usr/bin/env python
"""Plot Spearman's |rho| (OT distance vs. accuracy) vs. lambda for each combined embedding
f"{location}+bert" = lambda*location + (1-lambda)*bert, one line per location embedding.
Lambda=0/1 are the plain bert-only/location-only OT rows. Reads plot/plots/rho_overall.csv
(rerun plot/build_overall_rho_csv.py first if the lambda sweep has changed)."""

import pandas as pd
import matplotlib.pyplot as plt

from trend_common import EMBEDDING_LABELS, EMBEDDING_COLORS
from rho_csv_common import LOCATION_EMBEDDING_TYPES

DATASET = "geoyfcc_text"
rho_df = pd.read_csv("plot/plots/rho_overall.csv")
rho_df = rho_df[(rho_df["dataset"] == DATASET) & (rho_df["distance_type"] == "ot")]

fig, ax = plt.subplots(figsize=(5, 7))
bert_rho = rho_df.loc[rho_df["embedding_type"] == "bert", "abs_rho"].iloc[0]

for location in LOCATION_EMBEDDING_TYPES:
    combo = rho_df[rho_df["embedding_type"] == f"{location}+bert"].sort_values("lambda")
    location_rho = rho_df.loc[rho_df["embedding_type"] == location, "abs_rho"].iloc[0]
    lambdas = [0.0] + combo["lambda"].tolist() + [1.0]
    abs_rhos = [bert_rho] + combo["abs_rho"].tolist() + [location_rho]
    label = fr"$\lambda${EMBEDDING_LABELS[location]}+$(1-\lambda)${EMBEDDING_LABELS['bert']}"
    ax.plot(lambdas, abs_rhos, marker="o", markersize=5, linewidth=2, color=EMBEDDING_COLORS[location], label=label)

ax.set_xticks([i / 10 for i in range(11)])
ax.set_xticklabels([f"{i / 10:.1f}" if i in (0, 5, 10) else "" for i in range(11)])
ax.set_xlabel(r"Lambda ($\lambda$)", fontsize=20)
ax.set_ylabel(r"Rank Correlation with $\Delta_{D_s,D_t}$", fontsize=20)
ax.tick_params(axis="both", labelsize=16)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.35 * (ymax - ymin))
ax.legend(loc="upper right", fontsize=15, frameon=True)
fig.savefig("plot/plots/lambda_sweep_geoyfcc_text.png", bbox_inches="tight", dpi=300)
print("Saved plot to plot/plots/lambda_sweep_geoyfcc_text.png")

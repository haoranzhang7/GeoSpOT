#!/bin/bash
set -e

# Runs summarize_by_domain_trends.py with its defaults, aggregating every by_domain trend
# CSV under plot/plots/ into plot/plots/by_domain_trend_summary.csv.
#
# Usage:
#   ./plot/scripts/run_summarize_by_domain_trends.sh [other summarize_by_domain_trends.py args...]

python plot/summarize_by_domain_trends.py "$@"

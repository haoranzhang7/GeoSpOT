#!/usr/bin/env python
"""For each subset-selection log, extract the selected countries and save to CSV."""
import csv
import glob
import re

LOGS_ROOT = 'results/subset/logs'
OUT_CSV = 'plot/plots/selected_countries.csv'

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['log_file', 'selected_countries'])
    for log_file in sorted(glob.glob(f'{LOGS_ROOT}/**/*.log', recursive=True)):
        text = open(log_file, errors='ignore').read()
        m = re.search(r'Candidate domains \(K=\d+\):\s*\[(.*?)\]', text)
        if m:
            content = m.group(1).replace('np.int64(', '').replace(')', '')
            countries = re.findall(r'-?\d+', content)
            writer.writerow([log_file, ';'.join(countries)])

print(f'Saved to {OUT_CSV}')

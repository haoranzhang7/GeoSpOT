## GeoYFCC Text Data

This project expects the GeoYFCC text data to be available under:

```text
./data/geoyfcc/
```

### Expected files

`GeoYFCCText` looks for a metadata file in `./data/geoyfcc`:

- `geoyfcc_all_metadata_before_cleaning.csv` **or**
- `geoyfcc_all_metadata_before_cleaning.csv.parquet`

The loader will also create and reuse a filtered pickle for faster reloads:

- `geoyfcc_text_filtered_single_label.pkl`

You should also keep the original raw metadata and any additional columns (e.g. `country_id`, `label_id`, `split`) used by the code.

### How to obtain the data

The GeoYFCC metadata is publicly released by the authors of **“Adaptive Methods for Real-World Domain Generalization”** in the GeoYFCC repository [`abhimanyudubey/GeoYFCC`](https://github.com/abhimanyudubey/GeoYFCC).

To download the original metadata file via CLI (following their instructions):

```bash
pip install gdown
gdown http://drive.google.com/uc?id=1HvpAeEc37R9nLcI79iSeVCX2PYg3AgXZ
echo "db7419355b1e9827a2cf8f480ee36120  GeoYFCC.tar.gz" | md5sum -c -
```

You should see `OK` if the MD5 checksum matches. Then extract the archive and move the resulting metadata file to the expected location:

```bash
tar -xzf GeoYFCC.tar.gz
mkdir -p data/geoyfcc
mv GeoYFCC.pkl data/geoyfcc/
```

Next, convert the pickle metadata into CSV/Parquet using the provided utility:

```bash
python datasets/geoyfcc/utils/parse_files.py
```

This will read `GeoYFCC.pkl` from `data/geoyfcc/` and produce:

- `data/geoyfcc/geoyfcc_all_metadata_before_cleaning.csv`
- `data/geoyfcc/geoyfcc_all_metadata_before_cleaning.csv.parquet`

The loader will then:

- Load the CSV/Parquet metadata from `./data/geoyfcc`,
- Filter and save a single-label, text-filtered pickle (`geoyfcc_text_filtered_single_label.pkl`) for fast reuse.

For details about the original GeoYFCC dataset and its metadata schema, please refer to the official repository [`abhimanyudubey/GeoYFCC`](https://github.com/abhimanyudubey/GeoYFCC).

### Verifying the setup

From the repo root:

```bash
python datasets/geoyfcc/geoyfcc.py
```

This will try to load the GeoYFCCText dataset from `./data/geoyfcc` and print basic statistics. If the metadata file is missing or misnamed, you will get a clear `FileNotFoundError` indicating which path is expected.


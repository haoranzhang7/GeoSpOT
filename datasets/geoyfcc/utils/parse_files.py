import os
import pandas as pd


def load_pickle(pkl_path: str) -> pd.DataFrame:
    return pd.read_pickle(pkl_path)


def expand_yfcc_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "photoid",
        "uid",
        "unickname",
        "displayname",
        "datetaken",
        "dateuploaded",
        "capturedevice",
        "title",
        "description",
        "usertags",
        "machinetags",
        "longitude",
        "latitude",
        "accuracy",
        "pageurl",
        "downloadurl",
        "licensename",
        "licenseurl",
        "serverid",
        "farmid",
        "secret",
        "secretoriginal",
        "ext",
        "marker",
    ]

    metadata_expanded = df["yfcc_metadata"].str.split("\t", expand=True)
    metadata_expanded.columns = metadata_cols

    df_expanded = pd.concat([df.drop(columns=["yfcc_metadata"]), metadata_expanded], axis=1)

    numeric_cols = ["photoid", "accuracy", "serverid", "farmid", "marker"]
    for col in numeric_cols:
        df_expanded[col] = pd.to_numeric(df_expanded[col], errors="coerce")

    return df_expanded


def save_dataframe(df: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved expanded dataframe to:\n - {csv_path}\n - {parquet_path}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(repo_root, "data", "geoyfcc")

    candidates = ["GeoYFCC.pkl", "Geo-YFCC.pkl"]
    pkl_path = None
    for name in candidates:
        candidate_path = os.path.join(data_dir, name)
        if os.path.exists(candidate_path):
            pkl_path = candidate_path
            break

    if pkl_path is None:
        raise FileNotFoundError(
            f"GeoYFCC pickle not found in {data_dir}. "
            f"Expected one of: {', '.join(candidates)}"
        )

    csv_path = os.path.join(data_dir, "geoyfcc_all_metadata_before_cleaning.csv")
    parquet_path = os.path.join(data_dir, "geoyfcc_all_metadata_before_cleaning.csv.parquet")

    df = load_pickle(pkl_path)
    df_expanded = expand_yfcc_metadata(df)
    save_dataframe(df_expanded, csv_path, parquet_path)


if __name__ == "__main__":
    main()

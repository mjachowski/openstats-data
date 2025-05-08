from pathlib import Path

import polars as pl


def _norm_col_name(col_name: str) -> str:
    # Normalizes column names by replacing special characters with
    # underscores and converting to lowercase to ensure consistent
    # naming convention
    chars = [" / ", " ", "\n", "/"]
    for char in chars:
        col_name = col_name.strip().replace(char, "_")
    col_name.replace("(", "")
    col_name.replace(")", "")
    return col_name.lower()


def read_csv(
    fname_csv: str, cols: list[str], truncate_ragged_lines: bool = False
) -> pl.LazyFrame:
    """Read a csv file and return a polars lazy frame.

    Given an csv filename, preferentially read the feather file if
    it exists, or fall back to reading the csv file and save the
    feather file. Also subsets and normalizes the column names.

    Args:
        fname_csv (str): The csv filename.

    Returns:
        A polars DataFrame with the data
    """

    fname_feather = fname_csv.replace(".csv", ".feather").replace("*", "")

    if Path(fname_feather).is_file():
        lf = pl.scan_ipc(fname_feather)
    else:
        lf = pl.scan_csv(
            fname_csv,
            truncate_ragged_lines=truncate_ragged_lines,
            ignore_errors=True,
            infer_schema_length=None,
        )
        lf.sink_ipc(fname_feather)

    lf = pl.scan_csv(
        fname_csv,
        truncate_ragged_lines=truncate_ragged_lines,
        ignore_errors=True,
        infer_schema_length=None,
    )
    schema = lf.collect_schema()

    # Subset to columns of interest and make all column names lowercase.
    existing_cols = [
        col
        for col in schema
        if _norm_col_name(col) in cols or len(cols) == 0
    ]
    lf = lf.select(
        [
            pl.col(col).alias(_norm_col_name(col))
            for col in schema.names()
            if col in existing_cols
        ]
    )

    return lf

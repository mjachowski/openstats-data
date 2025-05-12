import polars as pl
import typer
from typing_extensions import Annotated

from github_permalink import get_current_permalink, github_permalink
from util import read_csv

app = typer.Typer()

############################################################
# Constants
############################################################
MAUI_ZIP_CODES = {
    96708: "Upcountry",  # Haiku
    96713: "East Maui",  # Hana
    96729: "Molokai",  # Hoolehua
    96732: "Central Maui",  # Kahului
    96748: "Molokai",  # Kaunakakai
    96753: "South Maui",  # Kihei
    96757: "Molokai",  # Kualapuu
    96761: "West Maui",  # Lahaina
    96763: "Lanai",  # Lanai City
    96767: "West Maui",  # Lahaina
    96768: "Upcountry",  # Makawao
    96770: "Molokai",  # Maunaloa
    96779: "Upcountry",  # Paia
    96784: "Central Maui",  # Puunene
    96788: "Upcountry",  # Pukalani
    96790: "Upcountry",  # Kula
    96793: "Central Maui",  # Wailuku
}

COL_MAP_2000 = {
    "name": "zcta",
    "h020001": "total",
    "h020002": "total_owner_occupied",
    "h020003": "total_owner_occupied_le_0.5",
    "h020004": "total_owner_occupied_0.51_1.0",
    "h020005": "total_owner_occupied_1.01_1.5",
    "h020006": "total_owner_occupied_1.51_2.0",
    "h020007": "total_owner_occupied_ge_2.01",
    "h020008": "total_renter_occupied",
    "h020009": "total_renter_occupied_le_0.5",
    "h020010": "total_renter_occupied_0.51_1.0",
    "h020011": "total_renter_occupied_1.01_1.5",
    "h020012": "total_renter_occupied_1.51_2.0",
    "h020013": "total_renter_occupied_ge_2.01",
}

COL_MAP_ACS = {
    "name": "zcta",
    "b25014_001e": "total",
    "b25014_002e": "total_owner_occupied",
    "b25014_003e": "total_owner_occupied_le_0.5",
    "b25014_004e": "total_owner_occupied_0.51_1.0",
    "b25014_005e": "total_owner_occupied_1.01_1.5",
    "b25014_006e": "total_owner_occupied_1.51_2.0",
    "b25014_007e": "total_owner_occupied_ge_2.01",
    "b25014_008e": "total_renter_occupied",
    "b25014_009e": "total_renter_occupied_le_0.5",
    "b25014_010e": "total_renter_occupied_0.51_1.0",
    "b25014_011e": "total_renter_occupied_1.01_1.5",
    "b25014_012e": "total_renter_occupied_1.51_2.0",
    "b25014_013e": "total_renter_occupied_ge_2.01",
}

COL_MAP_ACS_RACE = {
    "name": "zcta",
    "b25014a_001e": "total",
    "b25014a_002e": "total_le_1.00",
    "b25014a_003e": "total_ge_1.01",
}


############################################################
# Helpers
############################################################
def zcta_to_region(zcta: str) -> str:
    zip = int(zcta.split(" ")[1])
    return MAUI_ZIP_CODES[zip]


def load_occupancy_by_tenure(
    filename: str, col_map: dict[str, str], year: int
) -> pl.LazyFrame:
    # Read csv file
    cols = list(col_map.keys())
    lf = read_csv(filename, cols=cols, skip_rows_after_header=1)
    lf = lf.rename(col_map)

    # Filter to relevant zip codes
    zcta_list = [f"ZCTA5 {zip_code}" for zip_code in MAUI_ZIP_CODES.keys()]
    lf = lf.filter(pl.col("zcta").is_in(zcta_list))

    # Add region column and drop zcta column
    lf = lf.with_columns(
        pl.col("zcta")
        .map_elements(zcta_to_region, return_dtype=pl.Utf8)
        .alias("region")
    ).drop("zcta")

    # Aggregate by region
    lf = lf.group_by("region").sum()

    # Calculate crowded and very crowded percentages
    lf = (
        lf.with_columns(
            pl.col("total_owner_occupied_1.01_1.5")
            .add(pl.col("total_owner_occupied_1.51_2.0"))
            .add(pl.col("total_owner_occupied_ge_2.01"))
            .alias("total_owner_occupied_crowded"),
            pl.col("total_renter_occupied_1.01_1.5")
            .add(pl.col("total_renter_occupied_1.51_2.0"))
            .add(pl.col("total_renter_occupied_ge_2.01"))
            .alias("total_renter_occupied_crowded"),
        )
        .with_columns(
            pl.col("total_owner_occupied_crowded")
            .add(pl.col("total_renter_occupied_crowded"))
            .alias("total_crowded"),
        )
        .with_columns(
            pl.col("total_crowded")
            .truediv(pl.col("total"))
            .mul(100)
            .round(1)
            .alias("pct_crowded"),
        )
        .select(
            pl.lit(year).alias("year"),
            "region",
            "total_crowded",
            "total",
            "pct_crowded",
        )
        .sort("region")
    )

    return lf


def load_occupancy_for_race(
    filename: str, col_map: dict[str, str], year: int
) -> pl.LazyFrame:
    # Occupancy data by race is less detailed

    # Read csv file
    cols = list(col_map.keys())
    lf = read_csv(filename, cols=cols, skip_rows_after_header=1)
    lf = lf.rename(col_map)

    # Filter to relevant zip codes
    zcta_list = [f"ZCTA5 {zip_code}" for zip_code in MAUI_ZIP_CODES.keys()]
    lf = lf.filter(pl.col("zcta").is_in(zcta_list))

    # Add region column and drop zcta column
    lf = lf.with_columns(
        pl.col("zcta")
        .map_elements(zcta_to_region, return_dtype=pl.Utf8)
        .alias("region")
    ).drop("zcta")

    # Aggregate by region
    lf = lf.group_by("region").sum()

    # Calculate crowded percentages
    lf = (
        lf.with_columns(
            pl.col("total_ge_1.01").alias("total_crowded"),
            pl.col("total_ge_1.01")
            .truediv(pl.col("total"))
            .mul(100)
            .round(1)
            .alias("pct_crowded"),
        )
        .select(
            pl.lit(year).alias("year"),
            "region",
            "total_crowded",
            "total",
            "pct_crowded",
        )
        .sort("region")
    )

    return lf


############################################################
# Entrypoints
############################################################
_occupancy_2000_help = "Occupancy filename for 2000 Census (csv format)"
_occupancy_2011_help = (
    "Occupancy filename for 2011 American Community Survey (csv format)"
)
_occupancy_2015_help = (
    "Occupancy filename for 2015 American Community Survey (csv format)"
)
_occupancy_2020_help = (
    "Occupancy filename for 2020 American Community Survey (csv format)"
)
_occupancy_2023_help = (
    "Occupancy filename for 2023 American Community Survey (csv format)"
)
_occupancy_race_2011_help = "occupancy for specific race filename for 2011 American Community Survey (csv format)"
_occupancy_race_2015_help = "occupancy for specific race filename for 2015 American Community Survey (csv format)"
_occupancy_race_2020_help = "occupancy for specific race filename for 2020 American Community Survey (csv format)"
_occupancy_race_2023_help = "occupancy for specific race filename for 2023 American Community Survey (csv format)"
_out_help = "Output filename (csv format)"


@github_permalink
@app.command()
def crowding(
    occupancy_2000_filename: Annotated[
        str,
        typer.Option("--occupancy-2000", "-a", help=_occupancy_2000_help),
    ],
    occupancy_2011_filename: Annotated[
        str,
        typer.Option("--occupancy-2011", "-b", help=_occupancy_2011_help),
    ],
    occupancy_2015_filename: Annotated[
        str,
        typer.Option("--occupancy-2015", "-c", help=_occupancy_2015_help),
    ],
    occupancy_2020_filename: Annotated[
        str,
        typer.Option("--occupancy-2020", "-d", help=_occupancy_2020_help),
    ],
    occupancy_2023_filename: Annotated[
        str,
        typer.Option("--occupancy-2023", "-e", help=_occupancy_2023_help),
    ],
    occupancy_race_2011_filename: Annotated[
        str,
        typer.Option(
            "--occupancy-race-2011", "-f", help=_occupancy_race_2011_help
        ),
    ],
    occupancy_race_2015_filename: Annotated[
        str,
        typer.Option(
            "--occupancy-race-2015", "-g", help=_occupancy_race_2015_help
        ),
    ],
    occupancy_race_2020_filename: Annotated[
        str,
        typer.Option(
            "--occupancy-race-2020", "-i", help=_occupancy_race_2020_help
        ),
    ],
    occupancy_race_2023_filename: Annotated[
        str,
        typer.Option(
            "--occupancy-race-2023", "-j", help=_occupancy_race_2023_help
        ),
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    # Load data for all races
    # lf_2000 = load_occupancy_by_tenure(
    #     occupancy_2000_filename, COL_MAP_2000, 2000
    # )
    lf_2011 = load_occupancy_by_tenure(
        occupancy_2011_filename, COL_MAP_ACS, 2011
    )
    lf_2015 = load_occupancy_by_tenure(
        occupancy_2015_filename, COL_MAP_ACS, 2015
    )
    lf_2020 = load_occupancy_by_tenure(
        occupancy_2020_filename, COL_MAP_ACS, 2020
    )
    lf_2023 = load_occupancy_by_tenure(
        occupancy_2023_filename, COL_MAP_ACS, 2023
    )

    # Concatenate lazy frames
    lf_all_races = pl.concat(
        [lf_2011, lf_2015, lf_2020, lf_2023], how="vertical"
    )

    # Load data for specific race
    lf_race_2011 = load_occupancy_for_race(
        occupancy_race_2011_filename, COL_MAP_ACS_RACE, 2011
    )
    lf_race_2015 = load_occupancy_for_race(
        occupancy_race_2015_filename, COL_MAP_ACS_RACE, 2015
    )
    lf_race_2020 = load_occupancy_for_race(
        occupancy_race_2020_filename, COL_MAP_ACS_RACE, 2020
    )
    lf_race_2023 = load_occupancy_for_race(
        occupancy_race_2023_filename, COL_MAP_ACS_RACE, 2023
    )

    # Concatenate lazy frames
    lf_for_race = pl.concat(
        [lf_race_2011, lf_race_2015, lf_race_2020, lf_race_2023],
        how="vertical",
    )

    # Calculate lf for all data excluding specific race
    lf = lf_all_races.join(
        lf_for_race, on=["year", "region"], suffix="_race"
    )
    lf = lf.with_columns(
        pl.col("total_crowded")
        .sub(pl.col("total_crowded_race"))
        .alias("total_crowded_diff"),
        pl.col("total").sub(pl.col("total_race")).alias("total_diff"),
    ).with_columns(
        pl.col("total_crowded_diff")
        .truediv(pl.col("total_diff"))
        .mul(100)
        .round(1)
        .alias("pct_crowded_diff"),
    )

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

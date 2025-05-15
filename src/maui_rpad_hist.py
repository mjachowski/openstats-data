from functools import reduce

import polars as pl
import typer
from typing_extensions import Annotated

from github_permalink import get_current_permalink
from minatoya import MINATOYA_MAP, MINATOYA_TMKS
from util import read_csv

app = typer.Typer()

############################################################
# Constants
############################################################
NULL_CONST = "(null)"
MAX_YEAR = 2025

# Tax rate classes are defined by numbers in the file.
# These are string descriptions for what each number maps to.
TAX_RATE_CLASS_MAP = [
    "timeshare",  # 0
    "non-owner-occupied",  # 1
    "apartment",  # 2
    "commercial",  # 3
    "industrial",  # 4
    "agricultural",  # 5
    "conservation",  # 6
    "hotel/resort",  # 7
    "",  # 8
    "owner-occupied",  # 9
    "commercialized-res",  # 10
    "tvr-strh",  # 11
    "long-term-rental",  # 12
]


############################################################
# Helpers
############################################################
def add_maui_region_col(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Zone and section are the 2nd and 3rd digits in the TMK.
    # Zone and section neatly define the different regions of
    # Maui per the logic below.
    # See image M00000 in the Maui County Document Center:
    # Public Data Extracts > Tax Map Images > Zone 1
    lahaina_zs = [45, 46]
    west_maui_zs = [zs for zs in range(41, 49)]
    central_maui_zs = [zs for zs in range(31, 39)]
    upcountry_zs = [zs for zs in range(22, 30)]
    south_maui_zs = [21, 39]
    east_maui_zs = [zs for zs in range(11, 20)]
    molokai_zs = [zs for zs in range(50, 60)]
    lanai_zs = [49]

    lf = lf.with_columns(
        pl.col("tmk")
        .cast(pl.Utf8)
        .str.slice(1, 2)
        .cast(pl.Int64)
        .alias("zone_section")
    )

    # Create "is_{region}" boolean columns for each major region
    lf = lf.with_columns(
        pl.col("zone_section").is_in(lahaina_zs).alias("is_lahaina"),
        pl.col("zone_section").is_in(west_maui_zs).alias("is_west_maui"),
        pl.col("zone_section")
        .is_in(central_maui_zs)
        .alias("is_central_maui"),
        pl.col("zone_section").is_in(upcountry_zs).alias("is_upcountry"),
        pl.col("zone_section").is_in(south_maui_zs).alias("is_south_maui"),
        pl.col("zone_section").is_in(east_maui_zs).alias("is_east_maui"),
        pl.col("zone_section").is_in(molokai_zs).alias("is_molokai"),
        pl.col("zone_section").is_in(lanai_zs).alias("is_lanai"),
    )

    # Also create a single "region" column
    lf = lf.with_columns(
        pl.when(pl.col("is_lahaina"))
        .then(pl.lit("West Maui"))
        .when(pl.col("is_west_maui"))
        .then(pl.lit("West Maui"))
        .when(pl.col("is_central_maui"))
        .then(pl.lit("Central Maui"))
        .when(pl.col("is_upcountry"))
        .then(pl.lit("Upcountry"))
        .when(pl.col("is_south_maui"))
        .then(pl.lit("South Maui"))
        .when(pl.col("is_east_maui"))
        .then(pl.lit("East Maui"))
        .when(pl.col("is_molokai"))
        .then(pl.lit("Molokai"))
        .when(pl.col("is_lanai"))
        .then(pl.lit("Lanai"))
        .otherwise(None)
        .alias("region")
    )

    return lf


def get_complex_resident_occupancy(
    assessments_filename: str,
) -> pl.LazyFrame:
    # Read historical assessments file
    schema = {
        "parid": pl.Int64,
        "taxyr": pl.Int32,
        "valclass": pl.Utf8,
        "ovrclass": pl.Utf8,
        "cur": pl.Utf8,
    }
    cols = list(schema.keys())
    lf = read_csv(
        assessments_filename,
        cols=cols,
        infer_schema=False,
    )
    lf = lf.with_columns(
        [
            pl.col(col).cast(dtype, strict=False)
            for col, dtype in schema.items()
        ]
    )

    # Filter to relevant rows: cur=="Y" and either valclass or ovrclass
    # is not NULL_CONST
    lf = lf.filter(pl.col("cur").eq("Y")).filter(
        pl.col("valclass").ne(NULL_CONST)
        | pl.col("ovrclass").ne(NULL_CONST)
    )

    # Create tax_class_num column
    lf = lf.with_columns(
        pl.when(pl.col("ovrclass").ne(NULL_CONST))
        .then(pl.col("ovrclass"))
        .otherwise(pl.col("valclass"))
        .cast(pl.Int32)
        .alias("tax_class_num")
    )

    # Convert tax_class_num to tax_class
    lf = lf.with_columns(
        pl.col("tax_class_num")
        .map_elements(lambda n: TAX_RATE_CLASS_MAP[n], return_dtype=pl.Utf8)
        .alias("tax_class")
    )

    # Create tmk and tmk_sub (first 8 digits that identifies
    # property complexes) columns. Note that parid (parcel id) is not
    # exactly tmk - it is missing the leading '2', so we add it.
    lf = lf.with_columns(
        pl.lit(2).mul(1_000_000_000_000).add(pl.col("parid")).alias("tmk"),
        pl.col("parid")
        .map_elements(
            lambda parid: int(str(parid)[0:8]), return_dtype=pl.Int64
        )
        .alias("tmk_sub"),
    )

    # Drop unnecessary columns
    lf = lf.drop("parid", "ovrclass", "valclass", "cur", "tax_class_num")

    # Exclude current active year
    lf = lf.filter(pl.col("taxyr").le(MAX_YEAR))

    # Add Maui region column
    lf = add_maui_region_col(lf)

    # Filter to Minatoya properties
    lf = lf.filter(pl.col("tmk_sub").is_in(MINATOYA_TMKS))

    # Add Minatoya complex names
    lf = lf.with_columns(
        pl.col("tmk_sub").replace_strict(MINATOYA_MAP).alias("complex_name")
    )

    # Sort by tmk and taxyr
    lf = lf.sort("tmk", "taxyr")

    # Group by tmk_sub and year, calculate resident percent
    # Resident properties are owner-occupied or long-term-rental
    agg_lf = (
        lf.group_by("tmk_sub", "taxyr")
        .agg(
            pl.col("complex_name").first(),
            pl.col("region").first(),
            pl.len().alias("total_count"),
            pl.col("tax_class")
            .is_in(["owner-occupied", "long-term-rental"])
            .sum()
            .alias("res_count"),
        )
        .with_columns(
            pl.col("res_count")
            .truediv(pl.col("total_count"))
            .mul(100)
            .round(1)
            .alias("res_pct"),
        )
    ).sort("tmk_sub", "taxyr")

    # Calculate max owner occupied percent and other related
    # fields for each complex.
    summary_lf = (
        agg_lf.group_by("tmk_sub")
        .agg(
            pl.col("complex_name").first(),
            pl.col("region").first(),
            pl.col("res_count")
            .gather(pl.col("res_pct").arg_max())
            .alias("max_res_count"),
            pl.col("total_count")
            .gather(pl.col("res_pct").arg_max())
            .alias("total_count_at_max_res_pct"),
            pl.col("res_pct").max().alias("max_res_pct"),
            pl.col("taxyr")
            .gather(pl.col("res_pct").arg_max())
            .alias("taxyr_of_max_res_pct"),
            pl.col("total_count")
            .gather(pl.col("taxyr").arg_max())
            .alias("cur_total_count"),
            pl.col("res_pct")
            .gather(pl.col("taxyr").arg_max())
            .alias("cur_res_pct"),
        )
        .with_columns(
            pl.col("max_res_count").list.first().alias("max_res_count"),
            pl.col("total_count_at_max_res_pct")
            .list.first()
            .alias("total_count_at_max_res_pct"),
            pl.col("taxyr_of_max_res_pct")
            .list.first()
            .alias("taxyr_of_max_res_pct"),
            pl.col("cur_total_count").list.first().alias("cur_total_count"),
            pl.col("cur_res_pct").list.first().alias("cur_res_pct"),
        )
        .with_columns(
            pl.col("cur_res_pct").sub("max_res_pct").alias("res_pct_diff"),
        )
        .sort("max_res_pct", descending=True)
        .select(
            "complex_name",
            "tmk_sub",
            "region",
            "taxyr_of_max_res_pct",
            "max_res_count",
            "total_count_at_max_res_pct",
            "cur_total_count",
            "max_res_pct",
            "cur_res_pct",
            "res_pct_diff",
        )
    )

    return summary_lf


############################################################
# Entrypoints
############################################################
_count_type_help = "'units' or 'complexes'"
_assessments_help = "Assessments history csv file from Maui County RPAD"
_out_help = "Output filename (csv format)"


@app.command()
def minatoya_resident_occupancy(
    assessments_filename: Annotated[
        str, typer.Option("--assessments", "-a", help=_assessments_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    """TODO

    Args:
        assessments_filename (str): historical RPAD assessment data
        out_filename (str): Output filename (csv format)
    """

    lf = get_complex_resident_occupancy(assessments_filename)

    lf = lf.select(
        "complex_name",
        "tmk_sub",
        "region",
        "taxyr_of_max_res_pct",
        "max_res_count",
        "total_count_at_max_res_pct",
        "max_res_pct",
        "cur_res_pct",
        "res_pct_diff",
    )

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


@app.command()
def minatoya_thresh_counts(
    count_type: Annotated[
        str, typer.Option("--count-type", "-c", help=_count_type_help)
    ],
    assessments_filename: Annotated[
        str, typer.Option("--assessments", "-a", help=_assessments_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    lf = get_complex_resident_occupancy(assessments_filename)

    lf = lf.select(
        "complex_name",
        "tmk_sub",
        "region",
        "max_res_pct",
        "cur_total_count",
    )

    is_units = count_type == "units"

    thresh_list = list(range(0, 55, 5))

    lf_list = []
    for thresh in thresh_list:
        thresh_lf = (
            lf.filter(pl.col("max_res_pct").ge(thresh))
            .group_by("region")
            .agg(
                pl.col("cur_total_count").sum().alias(f"units_{thresh}%")
                if is_units
                else pl.len().alias(f"complexes_{thresh}%"),
            )
        )

        total_thresh_lf = lf.filter(
            pl.col("max_res_pct").ge(thresh)
        ).select(
            pl.lit("Maui County").alias("region"),
            pl.col("cur_total_count").sum().alias(f"units_{thresh}%")
            if is_units
            else pl.len().alias(f"complexes_{thresh}%"),
        )

        thresh_lf = pl.concat([thresh_lf, total_thresh_lf])
        lf_list.append(thresh_lf)

    lf = reduce(
        lambda left, right: left.join(right, on="region", how="left"),
        lf_list,
    )
    lf = lf.fill_null(0)
    lf = lf.sort(f"{count_type}_{thresh_list[0]}%", descending=True)

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

import polars as pl
import typer
from typing_extensions import Annotated

from github_permalink import get_current_permalink, github_permalink
from util import read_csv

app = typer.Typer()

############################################################
# Constants
############################################################
MAUI_ZIP_CODES = [
    96708,
    96713,
    96729,
    96732,
    96733,
    96742,
    96748,
    96753,
    96757,
    96761,
    96763,
    96767,
    96768,
    96770,
    96779,
    96784,
    96788,
    96790,
    96793,
]

# We have RPAD sales data through 2024, but NHGIS + FRED household
# income data only through 2023. So, put everything in 2023 dollars.
INFLATION_BASE_YEAR = 2023


############################################################
# Helpers
############################################################
def get_assessments_lf(
    filename_2024: str, filename_2023: str | None = None
) -> pl.LazyFrame:
    cols = ["tmk", "tax_rate_class"]
    lf = read_csv(filename_2024, cols=cols)

    # There are a few bad rows, get rid of them.
    lf = lf.filter(pl.col("tmk").cast(str).str.len_chars() == 13)

    # Assessments in 2024 are 0, so get them from another year.
    if filename_2023 is not None:
        cols = [
            "tmk",
            "assessed_building_value",
            "assessed_land_value",
            "building_exemption",
        ]
        lf2 = read_csv(filename_2023, cols=cols)

        # There are a few bad rows, get rid of them.
        lf2 = lf2.filter(pl.col("tmk").cast(str).str.len_chars() == 13)

        # Join tax class and assessment data.
        lf = lf.join(lf2, on="tmk", how="left")

    # Enrich and filter lazy frame.
    lf = add_maui_region_col(lf)
    lf = filter_to_residential(lf)

    return lf


def get_dwellings_lf(filename: str) -> pl.LazyFrame:
    cols = [
        "tmk",
        "sf_of_living_area",
        "year_built",
        "bed_rooms",
        "condo_type",
    ]
    lf = read_csv(filename, cols=cols)

    MIN_SQFT = 200
    lf = (
        # Add is_condo column
        lf.with_columns(pl.col("condo_type").gt(0).alias("is_condo"))
        .drop("condo_type")
        .with_columns(
            # Add home_type column ("Condo" or "Single Family Home")
            pl.when(pl.col("is_condo"))
            .then(pl.lit("Condo"))
            .otherwise(pl.lit("Single Family Home"))
            .alias("home_type"),
            # Ignore sf_of_living_area values that are suspiciously small
            pl.when(pl.col("sf_of_living_area").ge(MIN_SQFT))
            .then(pl.col("sf_of_living_area"))
            .otherwise(None),
        )
    )

    return lf


def aggregate_dwellings_by_tmk(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.sort(["tmk", "sf_of_living_area"], descending=[False, True])
        .group_by("tmk")
        .first()
    )


def get_owners_lf(filename: str) -> pl.LazyFrame:
    cols = ["tmk", "mailing_zip1", "country"]
    lf = read_csv(filename, cols=cols)

    lf = lf.group_by("tmk").agg(
        pl.col("mailing_zip1").max().alias("mailing_zip"),
        pl.col("mailing_zip1")
        .is_in(MAUI_ZIP_CODES)
        .any()
        .alias("is_maui_owner"),
    )

    return lf


def get_sales_lf(filename: str) -> pl.LazyFrame:
    cols = [
        "parid",
        "saledate",
        "recorddate",
        "price",
        "instruno",
        "landcourt_no",
        "cert_no",
    ]
    lf = read_csv(filename, cols=cols, truncate_ragged_lines=True)

    MIN_PRICE = 20000
    MIN_YEAR = 1984  # Very few sales before 1984 in record.
    MAX_YEAR = 2024

    # Convert parid to tmk.
    # Also filter out invalid prices and restrict time range.
    lf = (
        # Extract the year from saledate and recorddate, setting
        # invalid years to null.
        lf.with_columns(
            pl.col("saledate")
            .str.slice(0, 4)
            .cast(pl.Int64, strict=False)
            .alias("sale_year"),
            pl.col("recorddate")
            .str.slice(0, 4)
            .cast(pl.Int64, strict=False)
            .alias("record_year"),
        )
        .select(
            pl.concat_str([pl.lit("2"), pl.col("parid").cast(str)])
            .cast(pl.Int64)
            .alias("tmk"),
            # Sometimes sale year is 1900, which is clearly wrong.
            pl.when(pl.col("sale_year") == 1900)
            # In that case, replace it with record_year.
            .then(pl.col("record_year"))
            # If sale_year is null, replace it with record_year too.
            .otherwise(
                pl.coalesce(
                    [
                        pl.col("sale_year"),
                        pl.col("record_year"),
                    ]
                )
            )
            .alias("sale_year"),
            pl.col("price"),
            pl.col("saledate"),
            pl.col("recorddate"),
            pl.col("instruno"),
            pl.col("landcourt_no"),
            pl.col("cert_no"),
        )
        .filter(
            pl.col("price").gt(MIN_PRICE)
            & pl.col("sale_year").ge(MIN_YEAR)
            & pl.col("sale_year").le(MAX_YEAR)
        )
    )

    return lf


def get_cpi_lf(filename: str) -> pl.LazyFrame:
    cols = ["year", "annual"]
    lf = read_csv(filename, cols=cols)
    lf = lf.select(pl.col("year"), pl.col("annual").alias("cpi"))
    return lf


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


def filter_to_residential(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Ignore non-residential tax rate classes like timeshare,
    # commercial, conservation, etc. Also ignore really small
    # residential categories: apartment and commercialized-residential.
    residential_classes = [
        "owner-occupied",
        "non-owner-occupied",
        "tvr-strh",
        "long-term-rental",
    ]
    lf = lf.filter(pl.col("tax_rate_class").is_in(residential_classes))
    return lf


def get_assessments_dwellings_combined_lf(
    assessments_filename: str,
    dwellings_filename: str,
) -> pl.LazyFrame:
    # Get assessment data, this tells us which properties are residential.
    assessment_lf = get_assessments_lf(assessments_filename)
    assessment_lf = add_maui_region_col(assessment_lf)
    assessment_lf = filter_to_residential(assessment_lf)

    # Exactly 3 TMKs (out of over 64,000 residential TMKs) show up twice
    # in the assessment data because they are classified as having two
    # residential property tax classes, either short-term rental and
    # non-owner occupied, or owner-occupied and non-owner occupied.
    # The precise classification does not matter for analyzing property
    # sales, so we just keep the first one for each.
    assessment_lf = assessment_lf.group_by("tmk").first()

    # Get dwellings and just consider the largest dwelling for each TMK.
    dwellings_lf = get_dwellings_lf(dwellings_filename)
    dwellings_lf = aggregate_dwellings_by_tmk(dwellings_lf)

    # Inner join with dwellings to only consider properties with
    # dwellings, excluding undeveloped lots.
    lf = assessment_lf.join(dwellings_lf, on="tmk", how="inner")

    return lf


def get_combined_lf(
    assessments_filename_2024: str,
    assessments_filename_2023: str,
    dwellings_filename: str,
    owners_filename: str,
) -> pl.LazyFrame:
    # Load data from each file independently
    assessment_lf = get_assessments_lf(
        assessments_filename_2024, assessments_filename_2023
    )
    dwellings_lf = get_dwellings_lf(dwellings_filename)
    owners_lf = get_owners_lf(owners_filename)

    # Inner join with dwellings to only consider properties with dwellings
    lf = assessment_lf.join(dwellings_lf, on="tmk", how="inner")
    lf = add_ppsf_column(lf)

    # Left join with owners because we still want data for properties
    # without owner records. Add resident type columns.
    lf = lf.join(owners_lf, on="tmk", how="left")
    lf = add_resident_type_columns(lf)

    # Aggregate on tmk.
    # After joining, we have duplicate tmks because some properties
    # have multiple owners and/or multiple dewllings. So, we'll
    # take the first value for every column, except for dwelling and
    # owner information, where we will take the max values.
    all_cols = lf.collect_schema().keys()
    max_cols = ["sf_of_living_area", "ppsf", "is_maui_owner"]
    lf = lf.group_by("tmk").agg(
        *[pl.col(col).max() for col in max_cols],
        *[
            pl.col(col).first()
            for col in all_cols
            if col != "tmk" and col not in max_cols
        ],
    )

    return lf


def add_ppsf_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.when(pl.col("sf_of_living_area").gt(0))
        .then(
            pl.col("assessed_building_value")
            .truediv("sf_of_living_area")
            .round(0)
        )
        .otherwise(None)
        .alias("ppsf")
    )


def add_resident_type_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.when(
            pl.col("tax_rate_class").is_in(
                ["owner-occupied", "long-term-rental"]
            )
        )
        .then(pl.lit("Resident"))
        .otherwise(pl.lit("Non-Resident"))
        .alias("resident_type")
    )
    return lf


def adjust_for_inflation(
    lf: pl.LazyFrame, cpi_lf: pl.LazyFrame, col: str
) -> pl.LazyFrame:
    base_cpi = (
        cpi_lf.filter(pl.col("year").eq(INFLATION_BASE_YEAR))
        .select("cpi")
        .collect()
        .item()
    )

    lf = (
        lf.with_columns(pl.col("year").cast(pl.Int64))
        .join(cpi_lf, on="year", how="left")
        .with_columns(
            pl.col(col)
            .truediv("cpi")
            .mul(base_cpi)
            .round(0)
            .cast(pl.Int64)
            .alias(f"adj_{col}")
        )
        .drop("cpi")
    )
    return lf


@github_permalink
def property_sales(
    assessments_filename: str,
    dwellings_filename: str,
    sales_filename: str,
    cpi_filename: str,
    out_filename: str,
    is_condo: bool,
) -> None:
    # Get combined lazy frame of all residential properties with at
    # least once dwelling. Undeveloped lots are excluded.
    lf = get_assessments_dwellings_combined_lf(
        assessments_filename, dwellings_filename
    )

    # lf only has one row per tmk (property).
    # But, there could be multiple sales for each property, so
    # lf_with_sales will have multiple rows for each tmk that is
    # sold multiple times.
    # Note that inner join drops tmks with no sales.
    sales_lf = get_sales_lf(sales_filename)
    lf_with_sales = lf.join(sales_lf, on="tmk", how="inner")

    # Sometimes multiple parcels are sold at once, and they all
    # record as the same sale price. These are identified by rows
    # having the same saledate, price, recorddate, instruno,
    # landcourt_no, and cert_no. In this case, just keep one of
    # the sales because we only want to count the sale once.
    lf_with_sales = lf_with_sales.group_by(
        "saledate",
        "price",
        "recorddate",
        "instruno",
        "landcourt_no",
        "cert_no",
    ).first()

    # Create column with inflation adjusted sale price
    cpi_lf = get_cpi_lf(cpi_filename)
    lf_with_sales = lf_with_sales.with_columns(
        pl.col("sale_year").alias("year")
    ).drop("sale_year")
    lf_with_sales = adjust_for_inflation(lf_with_sales, cpi_lf, "price")

    # Filter to just single family homes or condos.
    # Get median sale price and sale count for each year in each region.
    lf_sales_summary = (
        lf_with_sales.filter(pl.col("is_condo").eq(is_condo))
        .group_by("region", "year")
        .agg(
            pl.col("price").median().cast(pl.Int64),
            pl.col("adj_price").median().cast(pl.Int64),
            pl.len().alias("count"),
        )
        .sort("region", "year")
    )

    # Write results
    df = lf_sales_summary.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


############################################################
# Entrypoints
############################################################
_assessments24_help = "Assessments (2024) csv file from Maui County RPAD"
_assessments23_help = "Assessments (2023) csv file from Maui County RPAD"
_dwellings_help = "Dwellings csv file from Maui County RPAD"
_owners_help = "Dwellings csv file from Maui County RPAD"
_sales_help = "Sales csv file from Maui County RPAD"
_cpi_help = "CPI csv file downloaded from US BLS"
_out_help = "Output filename (csv format)"


@app.command()
def single_family_home_sales(
    assessments_filename: Annotated[
        str, typer.Option("--assessments", "-a", help=_assessments24_help)
    ],
    dwellings_filename: Annotated[
        str, typer.Option("--dwellings", "-d", help=_dwellings_help)
    ],
    sales_filename: Annotated[
        str, typer.Option("--sales", "-s", help=_sales_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    """Calculate inflation-adjusted median single family home sale
    prices for different regions of Maui.

    Given Maui County Real Property Assessment Division (RPAD) data,
    calculate the median single family home sale price for each major
    region of Maui. Use Hawaii CPI data to adjust prices for inflation.

    Args:
        assessments_filename (str): RPAD assessment data
        dwellings_filename (str): RPAD dwellings data
        sales_filename (str): RPAD sales data
        cpi_filename (str): CPI for all items in urban Hawaii
        out_filename (str): Output filename (csv format)
    """

    property_sales(
        assessments_filename,
        dwellings_filename,
        sales_filename,
        cpi_filename,
        out_filename,
        is_condo=False,
    )


@app.command()
def condo_sales(
    assessments_filename: Annotated[
        str, typer.Option("--assessments", "-a", help=_assessments24_help)
    ],
    dwellings_filename: Annotated[
        str, typer.Option("--dwellings", "-d", help=_dwellings_help)
    ],
    sales_filename: Annotated[
        str, typer.Option("--sales", "-s", help=_sales_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    """Calculate inflation-adjusted median condo sale prices for
    different regions of Maui.

    Given Maui County Real Property Assessment Division (RPAD) data,
    calculate the median single family home sale price for each major
    region of Maui. Use Hawaii CPI data to adjust prices for inflation.

    Args:
        assessments_filename (str): RPAD assessment data
        dwellings_filename (str): RPAD dwellings data
        sales_filename (str): RPAD sales data
        cpi_filename (str): CPI for all items in urban Hawaii
        out_filename (str): Output filename (csv format)
    """

    property_sales(
        assessments_filename,
        dwellings_filename,
        sales_filename,
        cpi_filename,
        out_filename,
        is_condo=True,
    )


@github_permalink
@app.command()
def home_construction_by_decade(
    assessments_filename: Annotated[
        str, typer.Option("--assessments", "-a", help=_assessments24_help)
    ],
    dwellings_filename: Annotated[
        str, typer.Option("--dwellings", "-d", help=_dwellings_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    """Calculate home construction rates by decade for
    different regions of Maui.

    Args:
        assessments_filename (str): RPAD assessment data
        dwellings_filename (str): RPAD dwellings data
        out_filename (str): Output filename (csv format)
    """

    # Get combined lazy frame of all residential properties with at
    # least once dwelling. Undeveloped lots are excluded.
    lf = get_assessments_dwellings_combined_lf(
        assessments_filename, dwellings_filename
    )

    MIN_YEAR = 1970
    MAX_YEAR = 2023

    # Step 1: Filter to recent decades and add decade column
    base_lf = (
        lf.filter(pl.col("year_built").ge(MIN_YEAR))
        .filter(pl.col("year_built").le(MAX_YEAR))
        .with_columns(
            pl.col("year_built").floordiv(10).mul(10).alias("decade"),
        )
    )

    # Step 2: Calculate min and max years for each decade
    decade_ranges_lf = (
        base_lf.group_by("decade")
        .agg(
            pl.col("year_built").min().alias("decade_start"),
            pl.col("year_built").max().alias("decade_end"),
        )
        .with_columns(
            (
                pl.col("decade_start").cast(pl.Utf8)
                + "-"
                + pl.col("decade_end").cast(pl.Utf8)
            ).alias("decade_desc"),
            pl.col("decade_end")
            .sub("decade_start")
            .add(1)
            .alias("num_years"),
        )
    )

    # Step 3: Count homes by type within each decade and region
    counts_by_type_lf = base_lf.group_by(
        "region", "decade", "home_type"
    ).agg(pl.len().alias("decade_count"))

    # Create a complete set of all region, decade, home_type combinations
    # by creating a cartesian product of all possible values
    unique_regions = base_lf.select("region").unique()
    unique_decades = base_lf.select("decade").unique()
    unique_home_types = base_lf.select("home_type").unique()

    # Create cross joins to generate all possible combinations
    all_regions_decades = unique_regions.join(unique_decades, how="cross")
    all_combinations = all_regions_decades.join(
        unique_home_types, how="cross"
    )

    # Outer join with the counts to include missing combinations
    counts_by_type_lf = all_combinations.join(
        counts_by_type_lf, on=["region", "decade", "home_type"], how="left"
    ).with_columns(
        pl.col("decade_count").fill_null(0),
    )

    # Total counts across all regions by decade and home type
    counts_all_regions_lf = base_lf.group_by("decade", "home_type").agg(
        pl.len().alias("decade_count_all_regions")
    )
    counts_by_type_lf = counts_by_type_lf.join(
        counts_all_regions_lf, on=["decade", "home_type"], how="left"
    )

    # Step 4: Join the decade ranges with the counts
    joined_cols = [
        "home_type",
        "decade",
        "decade_desc",
        "num_years",
        "region",
        "decade_count",
        "decade_count_all_regions",
    ]
    joined_lf = (
        counts_by_type_lf.join(decade_ranges_lf, on="decade", how="left")
        .sort("home_type", "region", "decade")
        .select(joined_cols)
    )

    # Create a new home_type called "All Homes" that sums the
    # decade_count for each home_type, grouped by region and decade
    all_homes_lf = (
        joined_lf.group_by("region", "decade")
        .agg(
            pl.col("decade_count").sum().alias("decade_count"),
            pl.col("decade_count_all_regions")
            .sum()
            .alias("decade_count_all_regions"),
            pl.col("decade_desc").first(),
            pl.col("num_years").first(),
        )
        .with_columns(pl.lit("All Homes").alias("home_type"))
        .sort("home_type", "region", "decade")
        .select(joined_cols)
    )

    # Combine the original joined_lf with the new all_homes_lf
    joined_lf = pl.concat([joined_lf, all_homes_lf])

    # Step 5: Calculate decade yearly averages
    lf = joined_lf.with_columns(
        pl.col("decade_count")
        .floordiv(pl.col("num_years"))
        .alias("decade_yearly_avg"),
        pl.col("decade_count_all_regions")
        .floordiv(pl.col("num_years"))
        .alias("decade_yearly_avg_all_regions"),
        pl.col("decade_count")
        .truediv(pl.col("decade_count_all_regions"))
        .mul(100)
        .round(1)
        .alias("decade_region_pct"),
    ).drop("decade_count", "decade_count_all_regions", "num_years")

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


# @github_permalink
# @app.command()
# def dummy(
#     assessments_filename_2024: Annotated[
#         str, typer.Option("--assessments24", "-a", help=_assessments24_help)
#     ],
#     assessments_filename_2023: Annotated[
#         str, typer.Option("--assessments23", "-b", help=_assessments23_help)
#     ],
#     dwellings_filename: Annotated[
#         str, typer.Option("--dwellings", "-d", help=_dwellings_help)
#     ],
#     owners_filename: Annotated[
#         str, typer.Option("--owners", "-o", help=_owners_help)
#     ],
#     sales_filename: Annotated[
#         str, typer.Option("--sales", "-s", help=_sales_help)
#     ],
#     cpi_filename: Annotated[
#         str, typer.Option("--cpi", "-c", help=_cpi_help)
#     ],
# ) -> None:
#     """Calculate some statistic for Maui homes.
#
#     Describe the statistic here.
#
#     Args:
#         assessments_filename_2024 (str): RPAD assessment data (current)
#         assessments_filename_2023 (str): RPAD assessment data from 2023,
#             only used for building assessments, which are set to 0 for
#             many Lahaina properties in the current data
#         dwellings_filename (str): RPAD dwellings data (current)
#         owners_filename (str): RPAD owners data (current)
#         sales_filename (str): RPAD sales data (current)
#         cpi_filename (str): CPI for all items in urban Hawaii
#     """
#     pass


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

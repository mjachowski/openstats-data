import polars as pl
import typer
from typing_extensions import Annotated

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
    filename_2024: str, filename_2023: str
) -> pl.LazyFrame:
    cols = ["tmk", "tax_rate_class"]
    lf1 = read_csv(filename_2024, cols=cols)

    # Assessments in 2024 are 0, so get them from another year.
    cols = [
        "tmk",
        "assessed_building_value",
        "assessed_land_value",
        "building_exemption",
    ]
    lf2 = read_csv(filename_2023, cols=cols)

    # There are a few bad rows, get rid of them.
    lf1 = lf1.filter(pl.col("tmk").cast(str).str.len_chars() == 13)
    lf2 = lf2.filter(pl.col("tmk").cast(str).str.len_chars() == 13)

    # Join tax class and assessment data.
    lf = lf1.join(lf2, on="tmk", how="left")

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
    cols = ["parid", "saledate", "recorddate", "price"]
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


def property_sales(
    assessments_filename_2024: str,
    assessments_filename_2023: str,
    dwellings_filename: str,
    owners_filename: str,
    sales_filename: str,
    cpi_filename: str,
    is_condo: bool,
) -> None:
    # Join all of the data into a single lazy frame
    lf = get_combined_lf(
        assessments_filename_2024,
        assessments_filename_2023,
        dwellings_filename,
        owners_filename,
    )

    # lf will only have one row per tmk (property).
    # But, there could be multiple sales for each property, so
    # lf_with_sales will have multiple rows for each tmk that is
    # sold multiple times.
    # Note that inner join drops tmks with no sales.
    sales_lf = get_sales_lf(sales_filename)
    lf_with_sales = lf.join(sales_lf, on="tmk", how="inner")

    # Create column with inflation adjusted sale price
    cpi_lf = get_cpi_lf(cpi_filename)
    lf_with_sales = lf_with_sales.with_columns(
        pl.col("sale_year").alias("year")
    ).drop("sale_year")
    lf_with_sales = adjust_for_inflation(lf_with_sales, cpi_lf, "price")

    # Throwout sales with crazy prices (>= $20M). This only noticably
    # impacts the median sale price on Lanai in one year, where there
    # were some huge sales.
    MAX_PRICE = 20e6
    lf_with_sales = lf_with_sales.filter(pl.col("adj_price").lt(MAX_PRICE))

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
    print(df.write_csv(None))


############################################################
# Entrypoint
############################################################
_assessments24_help = (
    "Name of 2024 assessments csv file from Maui County RPAD"
)
_assessments23_help = (
    "Name of 2023 assessments csv file from Maui County RPAD"
)
_dwellings_help = "Name of dwellings csv file from Maui County RPAD"
_owners_help = "Name of dwellings csv file from Maui County RPAD"
_sales_help = "Name of sales csv file from Maui County RPAD"
_cpi_help = "Name of CPI csv file downloaded from US BLS"


@app.command()
def single_family_home_sales(
    assessments_filename_2024: Annotated[
        str, typer.Option("--assessments24", "-a", help=_assessments24_help)
    ],
    assessments_filename_2023: Annotated[
        str, typer.Option("--assessments23", "-b", help=_assessments23_help)
    ],
    dwellings_filename: Annotated[
        str, typer.Option("--dwellings", "-d", help=_dwellings_help)
    ],
    owners_filename: Annotated[
        str, typer.Option("--owners", "-o", help=_owners_help)
    ],
    sales_filename: Annotated[
        str, typer.Option("--sales", "-s", help=_sales_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
) -> None:
    property_sales(
        assessments_filename_2024,
        assessments_filename_2023,
        dwellings_filename,
        owners_filename,
        sales_filename,
        cpi_filename,
        is_condo=False,
    )


@app.command()
def condo_sales(
    assessments_filename_2024: Annotated[
        str, typer.Option("--assessments24", "-a", help=_assessments24_help)
    ],
    assessments_filename_2023: Annotated[
        str, typer.Option("--assessments23", "-b", help=_assessments23_help)
    ],
    dwellings_filename: Annotated[
        str, typer.Option("--dwellings", "-d", help=_dwellings_help)
    ],
    owners_filename: Annotated[
        str, typer.Option("--owners", "-o", help=_owners_help)
    ],
    sales_filename: Annotated[
        str, typer.Option("--sales", "-s", help=_sales_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
) -> None:
    property_sales(
        assessments_filename_2024,
        assessments_filename_2023,
        dwellings_filename,
        owners_filename,
        sales_filename,
        cpi_filename,
        is_condo=True,
    )


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

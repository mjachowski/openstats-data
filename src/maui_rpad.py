import polars as pl
import typer
from typing_extensions import Annotated

from github_permalink import get_current_permalink, github_permalink
from minatoya import MINATOYA_TMKS
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
    if filename_2023 is None:
        cols += [
            "assessed_building_value",
            "assessed_land_value",
            "building_exemption",
        ]

    lf = read_csv(filename_2024, cols=cols)
    lf = lf.with_columns(pl.col("tmk").cast(pl.Int64, strict=False))

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
        lf2 = lf2.with_columns(pl.col("tmk").cast(pl.Int64, strict=False))

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

    # Cast tmk column to int64.
    assessment_lf = assessment_lf.with_columns(
        pl.col("tmk").cast(pl.Int64, strict=False)
    )
    dwellings_lf = dwellings_lf.with_columns(
        pl.col("tmk").cast(pl.Int64, strict=False)
    )

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


def get_combined_lf_with_sales(
    assessments_filename: str,
    dwellings_filename: str,
    sales_filename: str,
    cpi_filename: str,
) -> pl.LazyFrame:
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

    return lf_with_sales


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
    lf_with_sales = get_combined_lf_with_sales(
        assessments_filename,
        dwellings_filename,
        sales_filename,
        cpi_filename,
    )

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

    # Now get median sale price and sale count for each year for county.
    county_lf_sales_summary = (
        lf_with_sales.filter(pl.col("is_condo").eq(is_condo))
        .group_by("year")
        .agg(
            pl.col("price").median().cast(pl.Int64),
            pl.col("adj_price").median().cast(pl.Int64),
            pl.len().alias("count"),
        )
        .select(
            pl.lit("Maui County").alias("region"),
            "year",
            "price",
            "adj_price",
            "count",
        )
        .sort("year")
    )

    lf_sales_summary = pl.concat(
        [lf_sales_summary, county_lf_sales_summary]
    )

    # Write results
    df = lf_sales_summary.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


def calc_condo_characteristics(lf: pl.LazyFrame, desc: str) -> pl.LazyFrame:
    total_count = lf.select(
        pl.lit(desc).alias("desc"), pl.len().alias("total_count")
    )
    total_count_lit = total_count.collect().item(0, 1)

    year_built = lf.select(
        pl.col("year_built")
        .median()
        .cast(pl.Int32)
        .alias("median_year_built"),
    )
    assessed_value = lf.filter(
        pl.col("assessed_building_value").gt(0)
    ).select(
        pl.col("assessed_building_value")
        .median()
        .cast(pl.Int64)
        .alias("median_assessed_building_value")
    )

    bedrooms = (
        lf.group_by("bed_rooms")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit(total_count_lit).alias("total_count"))
        .with_columns(
            pl.col("count")
            .truediv("total_count")
            .mul(100)
            .round(1)
            .alias("pct")
        )
    )

    br0 = bedrooms.filter(pl.col("bed_rooms").eq(0)).select(
        pl.col("pct").alias("pct_0br")
    )
    br1 = bedrooms.filter(pl.col("bed_rooms").eq(1)).select(
        pl.col("pct").alias("pct_1br")
    )
    br2 = bedrooms.filter(pl.col("bed_rooms").eq(2)).select(
        pl.col("pct").alias("pct_2br")
    )
    br3 = bedrooms.filter(pl.col("bed_rooms").eq(3)).select(
        pl.col("pct").alias("pct_3br")
    )

    tax_classes = (
        lf.group_by("tax_rate_class")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit(total_count_lit).alias("total_count"))
        .with_columns(
            pl.col("count")
            .truediv("total_count")
            .mul(100)
            .round(1)
            .alias("pct")
        )
    )

    tc_oo = tax_classes.filter(
        pl.col("tax_rate_class").eq("owner-occupied")
    ).select(pl.col("pct").alias("pct_oo"))
    tc_ltr = tax_classes.filter(
        pl.col("tax_rate_class").eq("long-term-rental")
    ).select(pl.col("pct").alias("pct_ltr"))
    tc_noo = tax_classes.filter(
        pl.col("tax_rate_class").eq("non-owner-occupied")
    ).select(pl.col("pct").alias("pct_noo"))
    tc_tvr = tax_classes.filter(
        pl.col("tax_rate_class").eq("tvr-strh")
    ).select(pl.col("pct").alias("pct_tvr"))

    result = pl.concat(
        [
            total_count,
            year_built,
            assessed_value,
            br0,
            br1,
            br2,
            br3,
            tc_oo,
            tc_ltr,
            tc_noo,
            tc_tvr,
        ],
        how="horizontal",
    )

    return result


############################################################
# Entrypoints
############################################################
_assessments24_help = "Assessments (2024) csv file from Maui County RPAD"
_assessments23_help = "Assessments (2023) csv file from Maui County RPAD"
_dwellings_help = "Dwellings csv file from Maui County RPAD"
_owners_help = "Dwellings csv file from Maui County RPAD"
_sales_help = "Sales csv file from Maui County RPAD"
_cpi_help = "CPI csv file downloaded from US BLS"
_hh_income_help = "Maui household income by region, generated by `nhgis.py maui-household-income`"
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
def affordable_sales(
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
    hh_income_filename: Annotated[
        str, typer.Option("--income", "-i", help=_hh_income_help)
    ],
    out_filename: Annotated[
        str, typer.Option("--out", "-o", help=_out_help)
    ],
) -> None:
    """Calculate the number of affordable property sales each year
    for different regions of Maui. Affordable is defined using the
    standard definition based on price to household income ratio:
    affordable sale prices are 5x the median household income or less.

    Args:
        assessments_filename (str): RPAD assessment data
        dwellings_filename (str): RPAD dwellings data
        sales_filename (str): RPAD sales data
        cpi_filename (str): CPI for all items in urban Hawaii
        hh_income_filename (str): Maui household income by region,
            generated by `nhgis.py maui-household-income ...`
        out_filename (str): Output filename (csv format)
    """

    # Get combined lazy frame of all residential properties with at
    # least once dwelling. Undeveloped lots are excluded.
    lf_with_sales = get_combined_lf_with_sales(
        assessments_filename,
        dwellings_filename,
        sales_filename,
        cpi_filename,
    )

    income_lf = read_csv(hh_income_filename, cols=[])

    # Add column for affordability threshold
    AFFORDABLE_PRICE_TO_INCOME_RATIO = 5
    income_lf = income_lf.with_columns(
        pl.col("median_household_income")
        .mul(AFFORDABLE_PRICE_TO_INCOME_RATIO)
        .alias("affordability_thresh")
    )

    # Join sales and income lazy frames
    lf = lf_with_sales.join(income_lf, on=["region", "year"], how="inner")

    # Calculate the count and percentage of affordable sales
    # for each home type, region, and year
    region_lf = (
        lf.group_by("home_type", "region", "year")
        .agg(
            pl.col("affordability_thresh").first(),
            pl.len().alias("total_count"),
            pl.col("price")
            .le(pl.col("affordability_thresh"))
            .sum()
            .alias("affordable_count"),
        )
        .with_columns(
            pl.col("affordable_count")
            .truediv("total_count")
            .mul(100)
            .round(1)
            .alias("pct_affordable")
        )
    ).sort(
        "home_type",
        "region",
        "year",
    )

    # Calculate county level stats too
    county_lf = (
        lf.group_by("home_type", "year")
        .agg(
            pl.col("affordability_thresh").first(),
            pl.len().alias("total_count"),
            pl.col("price")
            .le(pl.col("affordability_thresh"))
            .sum()
            .alias("affordable_count"),
        )
        .with_columns(
            pl.col("affordable_count")
            .truediv("total_count")
            .mul(100)
            .round(1)
            .alias("pct_affordable")
        )
        .select(
            "home_type",
            pl.lit("Maui County").alias("region"),
            "year",
            "affordability_thresh",
            "total_count",
            "affordable_count",
            "pct_affordable",
        )
    ).sort(
        "home_type",
        "region",
        "year",
    )

    # Join region and county data
    lf = pl.concat([region_lf, county_lf])

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


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
    ).agg(
        pl.len().alias("decade_count"),
        pl.col("sf_of_living_area").median().alias("sqft_median"),
    )

    # Get county counts as above, but do not group on region.
    county_counts_by_type_lf = base_lf.group_by("decade", "home_type").agg(
        pl.len().alias("decade_count"),
        pl.col("sf_of_living_area").median().alias("sqft_median"),
    )

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
    counts_all_regions_lf = county_counts_by_type_lf.select(
        pl.col("decade"),
        pl.col("home_type"),
        pl.col("decade_count").alias("decade_count_all_regions"),
        pl.col("sqft_median").alias("sqft_median_all_regions"),
    )
    counts_by_type_lf = counts_by_type_lf.join(
        counts_all_regions_lf, on=["decade", "home_type"], how="left"
    )
    county_counts_by_type_lf = county_counts_by_type_lf.join(
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
        "sqft_median",
        "sqft_median_all_regions",
    ]
    joined_lf = (
        counts_by_type_lf.join(decade_ranges_lf, on="decade", how="left")
        .sort("home_type", "region", "decade")
        .select(joined_cols)
    )
    county_joined_lf = (
        county_counts_by_type_lf.join(
            decade_ranges_lf, on="decade", how="left"
        )
        .with_columns(pl.lit("Maui County").alias("region"))
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
        .with_columns(
            pl.lit("All Homes").alias("home_type"),
            pl.lit(None).alias("sqft_median"),
            pl.lit(None).alias("sqft_median_all_regions"),
        )
        .sort("home_type", "region", "decade")
        .select(joined_cols)
    )

    county_all_homes_lf = (
        county_joined_lf.group_by("decade")
        .agg(
            pl.col("region").first(),
            pl.col("decade_count").sum().alias("decade_count"),
            pl.col("decade_count_all_regions")
            .sum()
            .alias("decade_count_all_regions"),
            pl.col("decade_desc").first(),
            pl.col("num_years").first(),
        )
        .with_columns(
            pl.lit("All Homes").alias("home_type"),
            pl.lit(None).alias("sqft_median"),
            pl.lit(None).alias("sqft_median_all_regions"),
        )
        .sort("home_type", "region", "decade")
        .select(joined_cols)
    )

    # Combine the per region data with the all homes and county data
    joined_lf = pl.concat(
        [joined_lf, all_homes_lf, county_joined_lf, county_all_homes_lf]
    )

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
        pl.col("sqft_median")
        .truediv(pl.col("sqft_median_all_regions"))
        .mul(100)
        .round(1)
        .alias("sqft_median_region_pct"),
    ).drop("decade_count", "decade_count_all_regions", "num_years")

    final_cols = [
        "home_type",
        "decade",
        "decade_desc",
        "region",
        "decade_yearly_avg",
        "decade_yearly_avg_all_regions",
        "decade_region_pct",
        "sqft_median",
        "sqft_median_all_regions",
        "sqft_median_region_pct",
    ]
    lf = lf.select(final_cols)

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


@github_permalink
@app.command()
def condo_characteristics(
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
    """Calculate condo characteristics (count, median year built,
    median assessed building value, percent of different number
    of bedrooms, percent of different tax classes) for different
    subsets of condos in West Maui, South Maui, and Central Maui.

    Args:
        assessments_filename (str): RPAD assessment data
        dwellings_filename (str): RPAD dwellings data
        out_filename (str): Output filename (csv format)
    """

    # Get combined lazy frame of all residential properties with at
    # Get combined lazy frame of all residential properties with at
    # least once dwelling. Undeveloped lots are excluded.
    lf = get_assessments_dwellings_combined_lf(
        assessments_filename, dwellings_filename
    )

    # Create tmk_sub column that uses first 9 digits of tmk
    # to identify property complexes.
    lf = lf.with_columns(
        pl.col("tmk")
        .map_elements(
            lambda tmk: int(str(tmk)[1:9]),
            return_dtype=pl.Int64,
        )
        .alias("tmk_sub")
    )

    # Extract Minatoya condos in West Maui and South Maui
    # (the vast majority of Minatoya properties).
    mlf = lf.filter(
        pl.col("tmk_sub").is_in(MINATOYA_TMKS)
        & pl.col("is_condo")
        & (pl.col("is_west_maui") | pl.col("is_south_maui"))
    )

    # Extract non-Minatoya condos in West Maui and South Maui
    # that are owner-occupied or long-term rentals.
    nmlf = lf.filter(
        ~pl.col("tmk_sub").is_in(MINATOYA_TMKS)
        & pl.col("is_condo")
        & (pl.col("is_west_maui") | pl.col("is_south_maui"))
        & pl.col("tax_rate_class").is_in(
            ["owner-occupied", "long-term-rental"]
        )
    )

    # Extract condos in Central Maui that are owner-occupied
    # long-term rentals.
    clf = lf.filter(
        pl.col("is_condo")
        & pl.col("is_central_maui")
        & pl.col("tax_rate_class").is_in(
            ["owner-occupied", "long-term-rental"]
        )
    )

    mlf_chars = calc_condo_characteristics(
        mlf, "west-south-maui-minatoya-condos"
    )
    nmlf_chars = calc_condo_characteristics(
        nmlf, "west-south-maui-non-minatoya-condos"
    )
    clf_chars = calc_condo_characteristics(clf, "central-maui-condos")

    lf = pl.concat([mlf_chars, nmlf_chars, clf_chars])

    # Write results
    df = lf.collect()
    df.write_csv(out_filename)

    # Write github permalink
    txt_filename = out_filename.replace(".csv", ".txt")
    with open(txt_filename, "w") as f:
        f.write(get_current_permalink() or "None")


@github_permalink
@app.command()
def new_home_usage(
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
    """Calculate number of recently constructed homes used
    by residents vs non-residents.

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

    MIN_YEAR = 2010
    MAX_YEAR = 2023

    # Step 1: Filter to recent decades
    base_lf = lf.filter(pl.col("year_built").ge(MIN_YEAR)).filter(
        pl.col("year_built").le(MAX_YEAR)
    )

    # Step 2: Add resident_type column
    resident_tax_classes = ["owner-occupied", "long-term-rental"]
    nonresident_tax_classes = ["non-owner-occupied", "tvr-strh"]
    base_lf = base_lf.with_columns(
        pl.when(pl.col("tax_rate_class").is_in(resident_tax_classes))
        .then(pl.lit("resident"))
        .when(pl.col("tax_rate_class").is_in(nonresident_tax_classes))
        .then(pl.lit("nonresident"))
        .otherwise(None)
        .alias("resident_type")
    )

    # Step 3: Count homes by home_type & resident_type in each region
    counts_by_type_lf = (
        base_lf.group_by("region", "home_type", "resident_type")
        .agg(
            pl.len().alias("count"),
        )
        .sort("home_type", "region", "resident_type")
    )

    # Get county counts as above, but do not group on region.
    county_counts_by_type_lf = (
        base_lf.group_by("home_type", "resident_type")
        .agg(
            pl.len().alias("count"),
        )
        .select(
            pl.lit("Maui County").alias("region"),
            "home_type",
            "resident_type",
            "count",
        )
        .sort("home_type", "region", "resident_type")
    )

    # Create a new home_type called "All Homes" that sums the
    # count for each home_type, grouped by region and resident_type
    all_homes_lf = (
        counts_by_type_lf.group_by("region", "resident_type")
        .agg(
            pl.col("count").sum().alias("count"),
        )
        .with_columns(
            pl.lit("All Homes").alias("home_type"),
        )
        .sort("home_type", "region", "resident_type")
    )

    county_all_homes_lf = (
        county_counts_by_type_lf.group_by("resident_type")
        .agg(
            pl.col("region").first(),
            pl.col("count").sum().alias("count"),
        )
        .with_columns(
            pl.lit("All Homes").alias("home_type"),
        )
        .select(
            "region",
            "resident_type",
            "count",
            "home_type",
        )
        .sort("home_type", "region", "resident_type")
    )

    # Combine the per region data with the county data
    join_cols = ["home_type", "region", "resident_type", "count"]
    joined_lf = pl.concat(
        [
            counts_by_type_lf.select(join_cols),
            county_counts_by_type_lf.select(join_cols),
            all_homes_lf.select(join_cols),
            county_all_homes_lf.select(join_cols),
        ]
    )

    # Add columns for region_count and pct
    total_lf = joined_lf.group_by("region", "home_type").agg(
        pl.col("count").sum().alias("region_count")
    )
    joined_lf = joined_lf.join(
        total_lf, on=["region", "home_type"], how="left"
    )
    joined_lf = joined_lf.with_columns(
        pl.col("count")
        .truediv(pl.col("region_count"))
        .mul(100)
        .round(1)
        .alias("pct")
    )

    lf = joined_lf.select(
        "home_type",
        "region",
        "resident_type",
        "count",
        "pct",
    )

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

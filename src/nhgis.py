import polars as pl
import typer
from typing_extensions import Annotated

from util import read_csv

app = typer.Typer()


############################################################
# Constants
############################################################
INCOME_INFLATION_BASE_YEAR = 2020

TRACT_TO_REGION_MAP = {
    30100: None,  # Hana
    30200: "Upcountry",  # Haiku
    30301: "Upcountry",  # Kula
    30302: "South Maui",  # Wailea (2000)
    30303: "South Maui",  # Wailea (2010)
    30304: "Upcountry",  # Omaopio
    30305: "Upcountry",  # Kanaio
    30306: "South Maui",  # Kula Makai
    30307: "South Maui",  # La Perouse
    30300: "Upcountry",  # Kula
    30400: "Upcountry",  # Makawao
    30500: "Upcountry",  # Paia
    30600: "Upcountry",  # Spreckelsville (2000)
    30700: "South Maui",  # Kihei
    30800: "Central Maui",  # Waihee-Waikapu
    30900: "Central Maui",  # Wailuku
    31000: "Central Maui",  # Wailuku
    31100: "Central Maui",  # Kahului
    31200: "Central Maui",  # Kahului
    31300: "Central Maui",  # Puunene
    31400: "West Maui",  # Lahaina
    31500: "West Maui",  # North West Maui
    31600: None,  # Lanai
    31700: None,  # Molokai
    31800: None,  # Molokai
    31900: "Upcountry",  # Spreckelsville, and Kalawao (?)
    32000: "West Maui",  # Launiopoko
}


############################################################
# Helpers
############################################################
def get_income_lf(filename: str) -> pl.LazyFrame:
    cols = ["year", "state", "tracta"]
    rename_cols = {"b79aa": "median_household_income"}
    cols += list(rename_cols.keys())

    lf = read_csv(filename, cols=cols)
    lf = lf.rename(rename_cols)
    lf = lf.filter(pl.col("state").eq("Hawaii")).drop("state")
    lf = normalize_acs_years(lf)
    lf = lf.with_columns(pl.col("median_household_income").cast(pl.Int32))
    return lf


def get_population_lf(filename: str) -> pl.LazyFrame:
    cols = ["year", "state", "tracta"]
    rename_cols = {"av0aa": "persons_total"}
    cols += list(rename_cols.keys())

    lf = read_csv(filename, cols=cols)
    lf = lf.rename(rename_cols)
    lf = lf.filter(pl.col("state").eq("Hawaii")).drop("state")
    lf = lf.with_columns(pl.col("persons_total").cast(pl.Int32))
    return lf


def get_cpi_lf(filename: str) -> pl.LazyFrame:
    cols = ["year", "annual"]
    lf = read_csv(filename, cols=cols)
    lf = lf.select(pl.col("year"), pl.col("annual").alias("cpi"))
    return lf


def normalize_acs_years(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Normal decennial years look like 1980, 1990, etc.
    # ACS years, look like "2011-2015". Normalize those years
    # to just have the final year, "2015" in the example case.
    has_range = pl.col("year").str.contains("-")
    lf = lf.with_columns(
        pl.when(has_range)
        # Extract the digits after the "-"
        .then(pl.col("year").str.extract(r"-(\d+)$"))
        .otherwise(pl.col("year"))
        .alias("year")
    )
    return lf


def get_combined_lf(
    income_lf: pl.LazyFrame, population_lf: pl.LazyFrame
) -> pl.LazyFrame:
    # Join lazy frames on year and census tract
    join_cols = ["year", "tracta"]
    lf = income_lf.join(population_lf, on=join_cols, how="left")
    return lf


def add_region_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Merge census tracts into regions
    lf = (
        lf.with_columns(
            pl.col("tracta").cast(pl.UInt32),
            pl.col("tracta")
            .cast(pl.UInt32)
            .floordiv(100)
            .mul(100)
            .alias("tract_base"),
        )
        .with_columns(
            pl.col("tracta")
            .replace_strict(TRACT_TO_REGION_MAP, default=None)
            .alias("region"),
            pl.col("tract_base")
            .replace_strict(TRACT_TO_REGION_MAP, default=None)
            .alias("tract_base_region"),
        )
        # Merge region and tract_base_region
        .with_columns(
            pl.col("region").fill_null(pl.col("tract_base_region"))
        )
        .filter(~pl.col("region").is_null())
        .drop("tract_base", "tract_base_region")
    )
    return lf


def adjust_for_inflation(
    lf: pl.LazyFrame, cpi_lf: pl.LazyFrame, col: str
) -> pl.LazyFrame:
    base_cpi = (
        cpi_lf.filter(pl.col("year").eq(INCOME_INFLATION_BASE_YEAR))
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


def aggregate_median_by_region(
    lf: pl.LazyFrame, median_col: str
) -> pl.LazyFrame:
    # We have medians in each census tract that we want to aggregate
    # into one median for the region. To do this, we will calculate
    # a weighted median, weighted by the population of each census tract.
    group_cols = ["year", "region"]
    weight_col = "persons_total"
    median_lf = (
        lf.select([*group_cols, weight_col, median_col])
        # Sort before calculating any cumsums
        .sort(group_cols + [median_col])
        # Calculate cumulative and total weights (population)
        .with_columns(
            pl.col(weight_col)
            .cum_sum()
            .over(group_cols)
            .alias("cumsum_weight"),
            pl.col(weight_col).sum().over(group_cols).alias("total_weight"),
        )
        # Find the row where cumulative weight is half of the total
        .with_columns(
            (
                (pl.col("cumsum_weight") - pl.col(weight_col) / 2)
                / pl.col("total_weight")
            )
            .over(group_cols)
            .alias("position")
        )
        .filter(pl.col("position").ge(0.5))
        # Select the first row after filtering, this is the median
        .group_by(group_cols)
        .agg(pl.col(median_col).first())
        .sort(group_cols)
    )
    return median_lf


############################################################
# Entrypoints
############################################################
_income_help = (
    "Name of household income csv file downloaded from IPUMS NHGIS"
)
_population_help = (
    "Name of household income csv file downloaded from IPUMS NHGIS"
)
_cpi_help = "Name of CPI csv file downloaded from US BLS"


@app.command()
def maui_household_income(
    income_filename: Annotated[
        str, typer.Option("--income", "-i", help=_income_help)
    ],
    population_filename: Annotated[
        str, typer.Option("--population", "-p", help=_population_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
) -> None:
    """Calculate inflation-adjusted median household income for different regions of Maui.

    Given NHGIS household income and population data at the census
    tract level, calculate the median household income for each
    major region of Maui. Use Hawaii CPI data to adjust incomes for
    inflation.

    Args:
        income_filename (str): NHGIS household income data, census tract level
        population_filename (str): NHGIS population data, census tract level
        cpi_filename (str): CPI for all items in ubran Hawaii

    Returns:
        Nothing. It prints the results as a csv to stdout.
    """

    income_lf = get_income_lf(income_filename)
    population_lf = get_population_lf(population_filename)
    cpi_lf = get_cpi_lf(cpi_filename)
    lf = get_combined_lf(income_lf, population_lf)

    lf = add_region_column(lf)
    lf = adjust_for_inflation(lf, cpi_lf, "median_household_income")
    lf = aggregate_median_by_region(lf, "adj_median_household_income")

    df = lf.collect()
    print(df.write_csv(None))


@app.command()
def dummy() -> None:
    print("Hello")


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

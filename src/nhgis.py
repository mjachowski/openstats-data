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


def get_fred_income_lf(filename: str) -> pl.LazyFrame:
    # Original column names are "mehoinushia646n" or "mhihi15009a052ncen",
    # but make life simpler but editing the file to rename whichever
    # column is there to "median_household_income"
    cols = ["observation_date", "median_household_income"]
    lf = read_csv(filename, cols=cols)

    lf = lf.select(
        pl.col("observation_date")
        .str.slice(0, 4)
        .cast(pl.Int64)
        .alias("year"),
        pl.col("median_household_income").cast(pl.Int64, strict=False),
    )

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


def get_combined_fred_income_lf(
    hawaii_lf: pl.LazyFrame, maui_lf: pl.LazyFrame
) -> pl.LazyFrame:
    # Hawaii data starts in 1984. Maui data starts in 1993, but has
    # some missing years. There are no missing years after 2000, so
    # create one continuous dataset that switches from State to Maui
    # starting in 2000.
    SWITCH_YEAR = 2000
    lf = pl.concat(
        [
            hawaii_lf.filter(pl.col("year").lt(SWITCH_YEAR)),
            maui_lf.filter(pl.col("year").ge(SWITCH_YEAR)),
        ]
    ).sort("year")
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


def interpolate_income_lf(
    lf: pl.LazyFrame, fred_lf: pl.LazyFrame, col: str
) -> pl.LazyFrame:
    # Create lazy frame with all years
    year = pl.col("year")
    minMax0 = lf.select(
        year.min().alias("min"), year.max().alias("max")
    ).collect()
    minMax1 = fred_lf.select(
        year.min().alias("min"), year.max().alias("max")
    ).collect()
    min_year = min(minMax0.item(0, 0), minMax1.item(0, 0))
    max_year = max(minMax0.item(0, 1), minMax1.item(0, 1))
    all_years = pl.DataFrame({"year": range(min_year, max_year + 1)}).lazy()

    # Get unique regions
    regions = lf.select("region").unique()

    # Create cross join of all years and regions
    all_years_and_regions = all_years.join(regions, how="cross")

    # Prepend mean 1980 data to fred_lf
    mean_1980_values_lf = (
        lf.filter(pl.col("year").eq(1980))
        .group_by("year")
        .agg(pl.col(col).median().round(0).cast(pl.Int64))
    )
    fred_plus_lf = pl.concat([mean_1980_values_lf, fred_lf])

    # Calculate ratio of each region to ference income at known points
    ratios = lf.join(fred_plus_lf, on="year", suffix="_ref").with_columns(
        pl.col(col)
        .truediv(pl.col(f"{col}_ref"))
        .alias("region_reference_ratio")
    )

    # Interpolate ratios for all years
    interpolated_ratios = (
        all_years_and_regions.join(
            ratios.select("year", "region", "region_reference_ratio"),
            on=["year", "region"],
            how="left",
        )
        .group_by("region")
        .agg(
            pl.col("year"),
            pl.col("region_reference_ratio")
            .interpolate()
            .alias("interpolated_ratio")
            .forward_fill()
            .backward_fill(),
        )
        .explode("year", "interpolated_ratio")
        .sort("region", "year")
    )

    # Join with reference data and calculate final interpolated values
    final_interpolated = (
        interpolated_ratios.join(fred_plus_lf, on="year", how="full")
        .with_columns(
            pl.col("interpolated_ratio")
            .mul(pl.col(col))
            .round(0)
            .cast(pl.Int64)
            .forward_fill()
            .alias("interpolated_income")
        )
        .select("year", "region", pl.col("interpolated_income").alias(col))
    )

    return final_interpolated


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
_fred_hawaii_income_help = (
    "Name of Hawaii household income csv file from FRED"
)
_fred_maui_income_help = (
    "Name of Maui estimated household income csv file from FRED"
)


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
        income_filename (str): NHGIS household income data,
            census tract level
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
def maui_household_income_interpolated(
    income_filename: Annotated[
        str, typer.Option("--income", "-i", help=_income_help)
    ],
    population_filename: Annotated[
        str, typer.Option("--population", "-p", help=_population_help)
    ],
    cpi_filename: Annotated[
        str, typer.Option("--cpi", "-c", help=_cpi_help)
    ],
    fred_hawaii_income_filename: Annotated[
        str,
        typer.Option(
            "--fred-hawaii-income", "-f", help=_fred_hawaii_income_help
        ),
    ],
    fred_maui_income_filename: Annotated[
        str,
        typer.Option(
            "--fred-hawaii-income", "-g", help=_fred_maui_income_help
        ),
    ],
) -> None:
    """Calculate inflation-adjusted median household income for different regions of Maui, using FRED household income data to interpolate between census years.

    Given NHGIS household income and population data at the census
    tract level, calculate the median household income for each
    major region of Maui. Income is only for census years (decennial),
    so we use annual data of Hawaii and Maui household incomes from FRED
    to interpolate region incomes between census years. Use Hawaii CPI
    data to adjust incomes for inflation.

    Args:
        income_filename (str): NHGIS household income data,
            census tract level
        population_filename (str): NHGIS population data, census tract level
        cpi_filename (str): CPI for all items in ubran Hawaii
        fred_hawaii_income_filename (str): Annual median household
            data for Hawaii
        fred_maui_income_filename (str): Annual estimated median
            household for Maui County

    Returns:
        Nothing. It prints the results as a csv to stdout.
    """

    income_lf = get_income_lf(income_filename)
    population_lf = get_population_lf(population_filename)
    cpi_lf = get_cpi_lf(cpi_filename)
    fred_hawaii_lf = get_fred_income_lf(fred_hawaii_income_filename)
    fred_maui_lf = get_fred_income_lf(fred_maui_income_filename)

    # Calculate inflation-adjusted income from census data only
    lf = get_combined_lf(income_lf, population_lf)
    lf = add_region_column(lf)
    lf = adjust_for_inflation(lf, cpi_lf, "median_household_income")
    lf = aggregate_median_by_region(lf, "adj_median_household_income")

    # Calculate reference inflation-adjusted income from fred data
    fred_lf = get_combined_fred_income_lf(fred_hawaii_lf, fred_maui_lf)
    fred_lf = adjust_for_inflation(
        fred_lf, cpi_lf, "median_household_income"
    ).drop("median_household_income")
    lf = interpolate_income_lf(lf, fred_lf, "adj_median_household_income")

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

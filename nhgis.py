import polars as pl
import typer

app = typer.Typer()


############################################################
# Constants
############################################################
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
def get_income_lf() -> pl.LazyFrame:
    cols = ["year", "state", "tracta"]
    rename_cols = {"b79aa": "median_household_income"}
    cols += list(rename_cols.keys())

    # TODO: implement
    lf = read_csv("filename.csv", cols=cols)
    lf = lf.rename(rename_cols)
    lf = lf.filter(pl.col("state").eq("Hawaii"))
    return lf


############################################################
# Entrypoint
############################################################
@app.command()
def household_income() -> None:
    income_lf = get_income_lf()
    # TODO


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

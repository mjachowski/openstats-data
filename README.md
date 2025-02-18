# Hawaii Open Stats Data

A collection of scripts to calculate statistics and generate csv
files from various input data.

## Recommended Setup

Make sure you have at least Python 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Setup virtual environment with uv (only once):
```bash
uv venv
uv pip install -r requirements.txt
```

Enter virtual environment:
```bash
source .venv/bin/activate
```

When you are done, exit your virtual environment:
```bash
deactivate
```

## Household Income
Get the data:

* Download NHGIS data from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org)
* Download CPI data from the [US Bureau of Labor Statistics (BLS)](https://www.bls.gov/cpi/data.htm)

The data sets are:

**NHGIS: Median Household Income in Previous Year**
* Selected years: 1980, 1990, 2000, 2006-2010, 2016-2020
* Code: B79
* Geographic level: Census Tract (by State--County)

**NHGIS: Total Population**
* Selected years: 1970, 1980, 1990, 2000, 2010, 2020
* Code: AV0
* Geographic level: Census Tract (by State--County)

**CPI: All items in Urban Hawaii, all urban consumers, not seasonablly adjusted**
* Series Id: CUURS49FSA0, CUUSS49FSA0
* Area: Urban Hawaii
* Item: All items
* Base Period: 1982-1984=100
* All years
* Annual Data


Generate inflation-adjusted household incomes for major regions of Maui:
```bash
# Get help message
python nhgis.py household-income --help

# Replace bracketed arguments with actual filename
python nhgis.py household-income -i [income.csv] -p [population.csv] -c [cpi.csv] > out/maui-household-income.csv
```

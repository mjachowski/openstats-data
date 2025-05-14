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

## Maui County Real Property Assessment Division (RPAD) data

Also known as Maui County Real Property Tax (RPT) data. Download the data from the [Maui County Document Center](https://www.mauicounty.gov/DocumentCenter/Index/231).

Interestingly, sales data is updated frequently (weekly?), but the rest of the property tax data is updated once per year in April.

The data is in a non-ideal fixed-width format. Convert it to csv format:
```bash
cd src/

# Get help message
python parse-maui-rpad.py parse-assessments --help

# Example command. Replace bracketed argument with actual filenames.
python parse-maui-rpad.py parse-assessments -f [raw_assessment_file] > out/assessments.csv
```

## Single Family Home and Condo Prices
Use the parsed RPAD data generated in the section above.

```bash
cd src/

# Example command. Replace bracketed argument with actual filenames.
python maui-rpad.py single-family-home-sales -a [assessments.csv] -d [dwellings.csv] -s [sales.csv] -c [cpi.csv] -o out/maui-sfh-sales.csv
```

## Single Family Home and Condo Construction by Decade
Use the parsed RPAD data generated in the section above.

```bash
cd src/

# Example command. Replace bracketed argument with actual filenames.
python maui-rpad.py home-construction-by-decade -a [assessments.csv] -d [dwellings.csv] -o out/maui-construction.csv
```

## Condo Characteristics
Use the parsed RPAD data generated in the section above.

```bash
cd src/

# Example command. Replace bracketed argument with actual filenames.
python maui_rpad.py condo-characteristics -a [assessments.csv] -d [dwellings.csv] -o out/maui-condo-characteristics.csv
```

## Affordable Sales
Use the parsed RPAD data generated in the section above and household
income data from the section below.

```bash
cd src/

# Example command. Replace bracketed argument with actual filenames.
python maui-rpad.py affordable-sales -a [assessments.csv] -d [dwellings.csv] -s [sales.csv] -c [cpi.csv] -i [income.csv] -o out/maui-affordable-sales.csv
```

## Household Income
Get the data:

* Download NHGIS data from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org)
* Download CPI data from the [US Bureau of Labor Statistics (BLS)](https://www.bls.gov/cpi/data.htm)

The data sets are:

**NHGIS: Median Household Income in Previous Year**
* Selected years: 1980, 1990, 2000, 2006-2010, 2011-2015, 2016-2020
* Code: B79
* Geographic level: Census Tract (by State--County)

**NHGIS: Median Household Income in Previous Year**
* Selected years: 1980, 1990, 2000, 2006-2010, 2011-2015, 2016-2020
* Code: B79
* Geographic level: County (by State)

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
cd src/

# Get help message
python nhgis.py household-income --help

# Replace bracketed arguments with actual filename
python nhgis.py household-income -i [income.csv] -j [county-income.csv] -p [population.csv] -c [cpi.csv] -o out/maui-household-income.csv
```

Generate derived affordability stats from above median sales data and household income data:
```bash
python derived.py maui-property-affordability -s [sales.csv] -i [income.csv] -o out/maui-affordability.csv
```

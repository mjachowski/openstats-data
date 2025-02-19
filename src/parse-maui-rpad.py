import typer
from typing_extensions import Annotated

app = typer.Typer()

############################################################
# Helpers
############################################################


def slices(line: str, *args: int) -> tuple[str, ...]:
    args += (len(line),)
    return tuple(
        line[args[i - 1] - 1 : args[i] - 1] for i in range(1, len(args))
    )


def sanitize_value(value: str) -> str:
    v = value.strip().lstrip("0")
    return v if len(v) > 0 else "0"


def sanitize(values: tuple[str, ...]) -> list[str]:
    return [sanitize_value(v) for v in values]


############################################################
# Entrypoints
############################################################


@app.command()
def parse_assessments(
    filename: Annotated[
        str,
        typer.Option(
            "--assessment-filename",
            "-f",
            help="Name of Maui County RPT Full Assessment File",
        ),
    ],
) -> None:
    """Convert raw Maui County Real Property Tax assessment data to csv.

    Raw Maui County Real Property Tax data is published to the
    Maui County Document Center every year in April. The data is
    in a fixed width format, where each field starts at a specific
    index in each line, as defined in a provided pdf file. This
    function parses the assessments file and writes out a csv.

    Args:
        filename (str): The raw data filename.

    Returns:
        Nothing. It prints the csv contents to stdout.
    """

    # Tax rate classes are defined by numbers in the file.
    # These are string descriptions for what each number maps to.
    tax_rate_class_map = [
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

    # The data file defines each field to start at a fixed index,
    # as defined in the pdf downloaded along with the data
    headers = (
        "tmk",
        "parcel_year",
        "land_class",
        "tax_rate_class",
        "assessed_land_value",
        "land_exemption",
        "assessed_building_value",
        "building_exemption",
    )
    starts = (1, 14, 19, 23, 27, 40, 53, 66)

    # Print csv headers
    print(",".join(headers))

    with open(filename, "r") as f:
        # Print each line as a csv row
        for line in f.readlines():
            s = sanitize(slices(line, *starts))
            # Replace numeric tax rate class with string description
            s[3] = tax_rate_class_map[int(s[3])]
            print(",".join(s))


@app.command()
def parse_dwellings(
    filename: Annotated[
        str,
        typer.Option(
            "--dwellings-filename",
            "-f",
            help="Name of Maui County RPT Full Dwellings File",
        ),
    ],
) -> None:
    """Convert raw Maui County Real Property Tax dwelling data to csv.

    Raw Maui County Real Property Tax data is published to the
    Maui County Document Center every year in April. The data is
    in a fixed width format, where each field starts at a specific
    index in each line, as defined in a provided pdf file. This
    function parses the dwellings file and writes out a csv.

    Args:
        filename (str): The raw data filename.

    Returns:
        Nothing. It prints the csv contents to stdout.
    """

    # The data file defines each field to start at a fixed index,
    # as defined in the pdf downloaded along with the data
    headers = (
        "tmk",
        "parcel_year",
        "card_number",
        "story_height",
        "exterior_wall",
        "framing",
        "style_occupancy",
        "roof_design",
        "full_baths",
        "half_baths",
        "additional_fixtures",
        "total_fixtures",
        "air_conditioning",
        "attic",
        "total_rooms",
        "bed_rooms",
        "family_rooms",
        "foundation",
        "basement",
        "construction",
        "flooring",
        "interior_wall_structure",
        "roof_material",
        "interior_wall_material",
        "condo_floor_level",
        "condo_type",
        "condo_view",
        "condo_parking_spaces",
        "condo_style",
        "duplex",
        "year_built",
        "effective_year_built",
        "physical_condition",
        "building_grade",
        "building_shape_factor",
        "percent_complete",
        "ceiling_material",
        "sf_of_living_area",
        "sf_of_bldg_foot_print",
        "addition_living_area_in_sf",
        "building_value",
        "built_in_fire_place_linear_feet",
        "no_prefab_fire_places",
        "no_wood_burning_fire_places",
        "cost_and_design_adjustment",
    )
    starts = (
        1,
        14,
        19,
        24,
        31,
        34,
        37,
        39,
        42,
        49,
        56,
        63,
        69,
        70,
        71,
        76,
        81,
        86,
        89,
        90,
        92,
        95,
        98,
        101,
        104,
        112,
        115,
        118,
        121,
        124,
        127,
        132,
        137,
        138,
        141,
        144,
        155,
        158,
        165,
        173,
        181,
        192,
        198,
        200,
        202,
    )

    # Print csv headers
    print(",".join(headers))

    with open(filename, "r") as f:
        # Print each line as a csv row
        for line in f.readlines():
            s = sanitize(slices(line, *starts))
            print(",".join(s))


@app.command()
def parse_owners(
    filename: Annotated[
        str,
        typer.Option(
            "--owner-filename",
            "-f",
            help="Name of Maui County RPT Full Owner File",
        ),
    ],
) -> None:
    """Convert raw Maui County Real Property Tax owner data to csv.

    Raw Maui County Real Property Tax data is published to the
    Maui County Document Center every year in April. The data is
    in a fixed width format, where each field starts at a specific
    index in each line, as defined in a provided pdf file. This
    function parses the owners file and writes out a csv.

    Args:
        filename (str): The raw data filename.

    Returns:
        Nothing. It prints the csv contents to stdout.
    """

    # The data file defines each field to start at a fixed index,
    # as defined in the pdf downloaded along with the data
    headers = (
        "tmk",
        "owner",
        "owner_type",
        "co_mailing_address",
        "mailing_street_address",
        "mailing_city_state_zip",
        "mailing_city_name",
        "mailing_state",
        "mailing_zip1",
        "mailing_zip2",
        "country",
    )
    starts = (1, 14, 54, 94, 215, 295, 387, 427, 429, 434, 438)

    # Print csv headers
    print(",".join(headers))

    with open(filename, "rb") as f:
        # Print each line as a csv row
        for bytes_line in f.readlines():
            # For some reason, this file is encoded weirdly
            line = bytes_line.decode("windows-1252")
            # Replace commas with | to play nice with csv output format
            line = line.replace(",", "|")
            s = sanitize(slices(line, *starts))
            print(",".join(s))


@app.command()
def parse_sales(
    filename: Annotated[
        str,
        typer.Option(
            "--sales-filename",
            "-f",
            help="Name of Maui County RPT Sales File",
        ),
    ],
) -> None:
    """Convert raw Maui County Real Property Tax sales data to csv.

    Raw Maui County Real Property Tax sales data is updated frequently
    in the Maui County Document Center. The data is in a fixed width
    , where each field starts at a specific index in each line, as
    defined in a provided pdf file. There are some other quirks too.
    This function parses the sales file and writes out a standard csv.

    Args:
        filename (str): The raw data filename.

    Returns:
        Nothing. It prints the csv contents to stdout.
    """

    # The data file defines each field to start at a fixed index,
    # as defined in the pdf downloaded along with the data.
    # Interestingly, this file contains headers, unlike the other
    # files, so we do not need to separately enumerate the headers.
    starts = (
        1,
        19,
        30,
        41,
        52,
        73,
        114,
        155,
        196,
        205,
        214,
        255,
        296,
        337,
        340,
    )

    with open(filename, "rb") as f:
        # Print each line as a csv row
        is_header = True
        for bytes_line in f.readlines():
            line = bytes_line.decode("windows-1252")
            line = line.strip()
            if len(line) == 0:
                continue
            # Print header as is, just lowercased
            if is_header:
                line = line.lower()
                is_header = False
            # For some reason header is repeated. Drop repeat headers
            # and footer row with "rows selected" text
            elif line.startswith("PARID") or "rows select" in line:
                continue
            s = sanitize(slices(line, *starts))
            # Funky format. Last character will be ",", delete that
            # and strip trailing whitespace
            s = [
                ss[:-1].strip()
                if len(ss) > 0 and ss[-1] == ","
                else ss.strip()
                for ss in s
            ]
            # Replace commas with | to play nice with csv output format
            s = [ss.replace(",", "|") for ss in s]
            print(",".join(s))


############################################################
# Main
############################################################
if __name__ == "__main__":
    app()

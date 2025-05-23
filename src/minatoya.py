# https://www.mauicounty.gov/DocumentCenter/View/112945/Short-Term-Occupancy-List-as-of-03202024
# Master TMKs with zoning A1 and A2 and allowed by ordinances.
# This excludes the leading digit (for county) and the final 4 digits
# (for unit & cpr info).
MINATOYA_TMKS = [
    # Hana
    14005040,  # HANA KAI-MAUI
    # Wailea
    21008060,  # WAILEA EKAHI II
    21008064,  # WAILEA EKAHI I
    21008065,  # WAILEA EKAHI III
    21008077,  # WAILEA EKOLU
    21008082,  # PALMS AT WAILEA I
    21008104,  # GRND CHAMP VILLAS
    # Paia
    26012050,  # KUAU PLAZA
    # Maalaea
    38014001,  # MAKANI A KAI
    38014002,  # HONO KAI
    38014004,  # KANAI A NALU
    38014011,  # MAALAEA BANYANS
    38014015,  # ISLAND SANDS
    38014016,  # LAULOA MAALAEA
    38014021,  # MAALAEA KAI
    38014022,  # MILOWAI-MAALAEA
    # Kihei
    39001002,  # MAUI SUNSET
    39001004,  # MAUI SCHOONER
    39001006,  # LUANA KAI
    39001057,  # WAIPUILANI
    39001075,  # KAUHALE MAKAI
    39001107,  # KIHEI BAY SURF
    39001110,  # LEINAALA
    39001134,  # KOA RESORT II
    39001136,  # KIHEI RESORT
    39001143,  # KIHEI BAY VISTA
    39004004,  # KAMAOLE SANDS
    39004081,  # MAUI HILL
    39004082,  # MAUI KAMAOLE III
    39004084,  # HALE KAMAOLE
    39004097,  # HALEAKALA SHORES
    39004098,  # MAUI PARKSHORE
    39004139,  # KEAWAKAPU
    39004143,  # MAUI KAMAOLE
    39004144,  # MAUI KAMAOLE II
    39005012,  # MY WAII BEACH COTTAGE
    39005013,  # WAILEA INN
    39005017,  # LIHIKAI APTS
    39005018,  # KIHEI COVE
    39005021,  # HALE MAHIALANI
    39005022,  # INDO LOTUS BEACH HSE
    39005023,  # KAMAOLE ONE
    39005035,  # HALE ILIILI
    39005038,  # PUNAHOA BEACH APTS
    39005039,  # 2131 ILIILI RD
    39007001,  # 1194 ULUNIU RD
    39007002,  # 1178 ULUNIU RD
    39007025,  # MOANA VILLA
    39008003,  # HALE KAI O'KIHEI
    39008009,  # LEILANI KAI
    39008011,  # KIHEI GARDEN ESTATES
    39008031,  # ALOHA VILLAS
    39009002,  # 1444 HALAMA ST
    39009003,  # 1440 HALAMA ST
    39009005,  # KAPU TOWNHOUSE
    39009010,  # WAIOHULI BCH DUPLEX
    39009025,  # 1470 HALAMA ST
    39009029,  # WAIOHULI BEACH HALE
    39016020,  # KIHEI VILLA
    39016027,  # KALAMA TERRACE
    39017003,  # SHORES OF MAUI
    39017010,  # KIHEI PARKSHORE
    39017017,  # KANOE APTS
    39018002,  # PACIFIC SHORES
    39018003,  # MAUI VISTA
    # Kapalua
    42001024,  # KAPALUA BAY VILLAS
    42001028,  # KAPALUA GOLF VILLAS
    42001032,  # THE RIDGE
    # Kahana
    43005009,  # KAHANA REEF
    43005020,  # KAHANA OUTRIGGER
    43005021,  # KAHANA OUTRIGGER
    43005029,  # KAHANA VILLAGE
    43005031,  # KAHANA OUTRIGGER
    43006006,  # LAHAINA BEACH CLUB
    43006007,  # NOHONANI
    43006012,  # MAKANI SANDS
    43006013,  # KALEIALOHA
    43006014,  # HONO KOA
    43006016,  # LOKELANI
    43006041,  # HALE MAHINA BEACH
    43006044,  # HALE ONO LOA
    43006063,  # PIKAKE
    43008001,  # MAHINAHINA BEACH
    43008002,  # POLYNESIAN SHORES
    43008004,  # KULEANA
    43008005,  # KULEANA
    43008006,  # HOYOCHI NIKKO
    43009002,  # NOELANI
    43009005,  # MAHINA SURF
    # Kaanapali
    44001041,  # HONOKOWAI PALMS
    44001042,  # HALE KAI I
    44001050,  # PAKI MAUI III
    44001051,  # PAKI MAUI I & II
    44001052,  # MAUI SANDS I
    44001055,  # PAPAKEA
    44001071,  # MAUI SANDS II
    44006011,  # HALE KAANAPALI
    44008021,  # MAUI ELDORADO
    44008023,  # KAANAPALI ROYAL
    # Lahaina
    45004002,  # PUUNOA BEACH ESTATES
    45013027,  # LAHAINA ROADS
    46010002,  # THE SPINNAKER
    # Molokai
    51003013,  # KENANI KAI
    56004055,  # WAVECREST
]

MINATOYA_MAP = {
    # Hana
    14005040: "HANA KAI-MAUI",
    # Wailea
    21008060: "WAILEA EKAHI II",
    21008064: "WAILEA EKAHI I",
    21008065: "WAILEA EKAHI III",
    21008077: "WAILEA EKOLU",
    21008082: "PALMS AT WAILEA I",
    21008104: "GRND CHAMP VILLAS",
    # Paia
    26012050: "KUAU PLAZA",
    # Maalaea
    38014001: "MAKANI A KAI",
    38014002: "HONO KAI",
    38014004: "KANAI A NALU",
    38014011: "MAALAEA BANYANS",
    38014015: "ISLAND SANDS",
    38014016: "LAULOA MAALAEA",
    38014021: "MAALAEA KAI",
    38014022: "MILOWAI-MAALAEA",
    # Kihei
    39001002: "MAUI SUNSET",
    39001004: "MAUI SCHOONER",
    39001006: "LUANA KAI",
    39001057: "WAIPUILANI",
    39001075: "KAUHALE MAKAI",
    39001107: "KIHEI BAY SURF",
    39001110: "LEINAALA",
    39001134: "KOA RESORT II",
    39001136: "KIHEI RESORT",
    39001143: "KIHEI BAY VISTA",
    39004004: "KAMAOLE SANDS",
    39004081: "MAUI HILL",
    39004082: "MAUI KAMAOLE III",
    39004084: "HALE KAMAOLE",
    39004097: "HALEAKALA SHORES",
    39004098: "MAUI PARKSHORE",
    39004139: "KEAWAKAPU",
    39004143: "MAUI KAMAOLE",
    39004144: "MAUI KAMAOLE II",
    39005012: "MY WAII BEACH COTTAGE",
    39005013: "WAILEA INN",
    39005017: "LIHIKAI APTS",
    39005018: "KIHEI COVE",
    39005021: "HALE MAHIALANI",
    39005022: "INDO LOTUS BEACH HSE",
    39005023: "KAMAOLE ONE",
    39005035: "HALE ILIILI",
    39005038: "PUNAHOA BEACH APTS",
    39005039: "2131 ILIILI RD",
    39007001: "1194 ULUNIU RD",
    39007002: "1178 ULUNIU RD",
    39007025: "MOANA VILLA",
    39008003: "HALE KAI O'KIHEI",
    39008009: "LEILANI KAI",
    39008011: "KIHEI GARDEN ESTATES",
    39008031: "ALOHA VILLAS",
    39009002: "1444 HALAMA ST",
    39009003: "1440 HALAMA ST",
    39009005: "KAPU TOWNHOUSE",
    39009010: "WAIOHULI BCH DUPLEX",
    39009025: "1470 HALAMA ST",
    39009029: "WAIOHULI BEACH HALE",
    39016020: "KIHEI VILLA",
    39016027: "KALAMA TERRACE",
    39017003: "SHORES OF MAUI",
    39017010: "KIHEI PARKSHORE",
    39017017: "KANOE APTS",
    39018002: "PACIFIC SHORES",
    39018003: "MAUI VISTA",
    # Kapalua
    42001024: "KAPALUA BAY VILLAS",
    42001028: "KAPALUA GOLF VILLAS",
    42001032: "THE RIDGE",
    # Kahana
    43005009: "KAHANA REEF",
    43005020: "KAHANA OUTRIGGER",
    43005021: "KAHANA OUTRIGGER",
    43005029: "KAHANA VILLAGE",
    43005031: "KAHANA OUTRIGGER",
    43006006: "LAHAINA BEACH CLUB",
    43006007: "NOHONANI",
    43006012: "MAKANI SANDS",
    43006013: "KALEIALOHA",
    43006014: "HONO KOA",
    43006016: "LOKELANI",
    43006041: "HALE MAHINA BEACH",
    43006044: "HALE ONO LOA",
    43006063: "PIKAKE",
    43008001: "MAHINAHINA BEACH",
    43008002: "POLYNESIAN SHORES",
    43008004: "KULEANA",
    43008005: "KULEANA",
    43008006: "HOYOCHI NIKKO",
    43009002: "NOELANI",
    43009005: "MAHINA SURF",
    # Kaanapali
    44001041: "HONOKOWAI PALMS",
    44001042: "HALE KAI I",
    44001050: "PAKI MAUI III",
    44001051: "PAKI MAUI I & II",
    44001052: "MAUI SANDS I",
    44001055: "PAPAKEA",
    44001071: "MAUI SANDS II",
    44006011: "HALE KAANAPALI",
    44008021: "MAUI ELDORADO",
    44008023: "KAANAPALI ROYAL",
    # Lahaina
    45004002: "PUUNOA BEACH ESTATES",
    45013027: "LAHAINA ROADS",
    46010002: "THE SPINNAKER",
    # Molokai
    51003013: "KENANI KAI",
    56004055: "WAVECREST",
}

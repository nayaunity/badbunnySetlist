"""Configuration and constants for setlist data collection."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
SETLISTFM_API_KEY = os.getenv("SETLISTFM_API_KEY")
SETLISTFM_BASE_URL = "https://api.setlist.fm/rest/1.0"

# Bad Bunny Configuration
BAD_BUNNY_MBID = "3d2dfd1c-8871-475f-ab64-6f07e7d32c5c"

# Puerto Rico Residency (2024-2025) - High Priority
PUERTO_RICO_RESIDENCY = {
    "venue_keywords": ["coliseo", "puerto rico", "san juan"],
    "date_range": {
        "start": "01-01-2024",  # dd-MM-yyyy format
        "end": "31-12-2025"
    },
    "priority": "high"
}

# Super Bowl Halftime Shows (2015-2025)
SUPERBOWL_HALFTIME_DATA = [
    {
        "year": 2025,
        "date": "09-02-2025",
        "artists": ["Kendrick Lamar"],
        "venue": "Caesars Superdome",
        "city": "New Orleans",
        "super_bowl": "LIX"
    },
    {
        "year": 2024,
        "date": "11-02-2024",
        "artists": ["Usher"],
        "venue": "Allegiant Stadium",
        "city": "Las Vegas",
        "super_bowl": "LVIII"
    },
    {
        "year": 2023,
        "date": "12-02-2023",
        "artists": ["Rihanna"],
        "venue": "State Farm Stadium",
        "city": "Glendale",
        "super_bowl": "LVII"
    },
    {
        "year": 2022,
        "date": "13-02-2022",
        "artists": ["Dr. Dre", "Snoop Dogg", "Eminem", "Mary J. Blige", "Kendrick Lamar"],
        "venue": "SoFi Stadium",
        "city": "Inglewood",
        "super_bowl": "LVI"
    },
    {
        "year": 2021,
        "date": "07-02-2021",
        "artists": ["The Weeknd"],
        "venue": "Raymond James Stadium",
        "city": "Tampa",
        "super_bowl": "LV"
    },
    {
        "year": 2020,
        "date": "02-02-2020",
        "artists": ["Shakira", "Jennifer Lopez"],
        "venue": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "super_bowl": "LIV"
    },
    {
        "year": 2019,
        "date": "03-02-2019",
        "artists": ["Maroon 5"],
        "venue": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "super_bowl": "LIII"
    },
    {
        "year": 2018,
        "date": "04-02-2018",
        "artists": ["Justin Timberlake"],
        "venue": "U.S. Bank Stadium",
        "city": "Minneapolis",
        "super_bowl": "LII"
    },
    {
        "year": 2017,
        "date": "05-02-2017",
        "artists": ["Lady Gaga"],
        "venue": "NRG Stadium",
        "city": "Houston",
        "super_bowl": "LI"
    },
    {
        "year": 2016,
        "date": "07-02-2016",
        "artists": ["Coldplay", "Beyonce", "Bruno Mars"],
        "venue": "Levi's Stadium",
        "city": "Santa Clara",
        "super_bowl": "50"
    },
    {
        "year": 2015,
        "date": "01-02-2015",
        "artists": ["Katy Perry"],
        "venue": "University of Phoenix Stadium",
        "city": "Glendale",
        "super_bowl": "XLIX"
    }
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Rate Limiting
REQUESTS_PER_MINUTE = 10
REQUEST_DELAY_SECONDS = 6

# Pagination
ITEMS_PER_PAGE = 20

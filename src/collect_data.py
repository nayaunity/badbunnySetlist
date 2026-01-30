"""Main entry point for setlist data collection."""
import json
import logging
from datetime import datetime
from pathlib import Path

from .config import DATA_RAW_DIR, SETLISTFM_API_KEY
from .api_client import SetlistFMClient
from .collectors.bad_bunny import collect_bad_bunny_setlists
from .collectors.superbowl import collect_superbowl_halftime_setlists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_json(data: dict, filepath: Path) -> None:
    """Save data as JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {filepath}")


def generate_report(bad_bunny_data: dict, superbowl_data: dict) -> dict:
    """Generate a summary report of the collection."""
    bb_songs = set()
    bb_venues = set()
    bb_tours = set()

    for setlist in bad_bunny_data.get("setlists", []):
        for song in setlist.get("songs", []):
            bb_songs.add(song["name"])
        venue = setlist.get("venue", {})
        if venue.get("name"):
            bb_venues.add(venue["name"])
        if setlist.get("tour"):
            bb_tours.add(setlist["tour"])

    sb_songs_list = []
    for show in superbowl_data.get("shows", []):
        if show.get("setlist_found"):
            sb_songs_list.append(show["song_count"])

    sb_avg_songs = sum(sb_songs_list) / len(sb_songs_list) if sb_songs_list else 0

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "bad_bunny": {
            "total_setlists": bad_bunny_data.get("total_setlists", 0),
            "unique_songs": len(bb_songs),
            "unique_venues": len(bb_venues),
            "unique_tours": len(bb_tours),
            "puerto_rico_residency_shows": bad_bunny_data.get("puerto_rico_residency_count", 0),
            "date_range": {
                "earliest": bad_bunny_data["setlists"][-1]["event_date"] if bad_bunny_data.get("setlists") else None,
                "latest": bad_bunny_data["setlists"][0]["event_date"] if bad_bunny_data.get("setlists") else None
            }
        },
        "super_bowl_halftime": {
            "total_shows": superbowl_data.get("total_shows", 0),
            "setlists_found": superbowl_data.get("setlists_found", 0),
            "setlists_missing": superbowl_data.get("setlists_missing", 0),
            "average_songs_per_show": round(sb_avg_songs, 1),
            "song_count_range": {
                "min": min(sb_songs_list) if sb_songs_list else 0,
                "max": max(sb_songs_list) if sb_songs_list else 0
            }
        },
        "collection_status": "complete" if bad_bunny_data.get("setlists") else "failed"
    }

    return report


def main():
    """Main function to run data collection."""
    logger.info("=" * 60)
    logger.info("Bad Bunny Super Bowl Setlist Prediction - Data Collection")
    logger.info("=" * 60)

    # Verify API key
    if not SETLISTFM_API_KEY or SETLISTFM_API_KEY == "your_api_key_here":
        logger.error("SETLISTFM_API_KEY not found or not set!")
        logger.error("Please update your .env file with your API key.")
        logger.error("Get your key at: https://www.setlist.fm/settings/api")
        return

    # Ensure output directory exists
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize API client
    client = SetlistFMClient()

    # Collect Bad Bunny setlists
    logger.info("\n" + "=" * 40)
    logger.info("DATASET 1: Bad Bunny Setlists")
    logger.info("=" * 40)
    bad_bunny_data = collect_bad_bunny_setlists(client)
    save_json(bad_bunny_data, DATA_RAW_DIR / "bad_bunny_setlists.json")

    # Collect Super Bowl halftime setlists
    logger.info("\n" + "=" * 40)
    logger.info("DATASET 2: Super Bowl Halftime Shows")
    logger.info("=" * 40)
    superbowl_data = collect_superbowl_halftime_setlists(client)
    save_json(superbowl_data, DATA_RAW_DIR / "superbowl_halftime_setlists.json")

    # Generate and save report
    logger.info("\n" + "=" * 40)
    logger.info("Generating Collection Report")
    logger.info("=" * 40)
    report = generate_report(bad_bunny_data, superbowl_data)
    save_json(report, DATA_RAW_DIR / "collection_report.json")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Bad Bunny setlists collected: {report['bad_bunny']['total_setlists']}")
    logger.info(f"Bad Bunny unique songs: {report['bad_bunny']['unique_songs']}")
    logger.info(f"Puerto Rico residency shows: {report['bad_bunny']['puerto_rico_residency_shows']}")
    logger.info(f"Super Bowl setlists found: {report['super_bowl_halftime']['setlists_found']}/{report['super_bowl_halftime']['total_shows']}")
    logger.info(f"Average songs per halftime show: {report['super_bowl_halftime']['average_songs_per_show']}")
    logger.info("=" * 60)
    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()

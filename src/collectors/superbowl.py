"""Collector for Super Bowl halftime show setlist data."""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..api_client import SetlistFMClient
from ..config import SUPERBOWL_HALFTIME_DATA

logger = logging.getLogger(__name__)


def find_superbowl_setlist(
    client: SetlistFMClient,
    artist_name: str,
    date: str,
    venue_name: str,
    city_name: str
) -> Optional[Dict[str, Any]]:
    """Find a Super Bowl halftime setlist by searching artist and matching date.

    Approach: Search by artist name, then find the setlist matching the Super Bowl date.
    This is more reliable than searching by exact date or venue.
    """
    year = int(date.split("-")[2])

    # Search by artist name and year, then look for the Super Bowl date
    logger.info(f"Searching for {artist_name} setlists in {year}...")

    # Check multiple pages since the Super Bowl setlist might not be on page 1
    for page in range(1, 4):  # Check up to 3 pages
        response = client.search_setlists(artist_name=artist_name, year=year, page=page)

        setlists = response.get("setlist", [])
        if not setlists:
            break

        for setlist in setlists:
            event_date = setlist.get("eventDate", "")
            if event_date == date:
                logger.info(f"Found {artist_name} setlist on {date} (page {page})")
                return setlist

        # If we got fewer than 20 results, no more pages
        if len(setlists) < 20:
            break

    # Fallback: try searching all setlists for the artist (no year filter)
    # and look through recent pages
    logger.info(f"Trying broader search for {artist_name}...")
    for page in range(1, 6):  # Check up to 5 pages
        response = client.search_setlists(artist_name=artist_name, page=page)

        setlists = response.get("setlist", [])
        if not setlists:
            break

        for setlist in setlists:
            event_date = setlist.get("eventDate", "")
            if event_date == date:
                logger.info(f"Found {artist_name} setlist on {date} (broader search, page {page})")
                return setlist

        if len(setlists) < 20:
            break

    logger.warning(f"No setlist found for {artist_name} on {date}")
    return None


def extract_halftime_data(
    setlist: Optional[Dict[str, Any]],
    sb_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract and normalize halftime show data."""
    result: Dict[str, Any] = {
        "super_bowl": sb_info["super_bowl"],
        "year": sb_info["year"],
        "date": sb_info["date"],
        "venue": sb_info["venue"],
        "city": sb_info["city"],
        "headliner": sb_info["artists"][0],
        "all_artists": sb_info["artists"],
        "setlist_found": setlist is not None,
        "songs": [],
        "song_count": 0,
        "guest_artists": []
    }

    if setlist:
        result["setlist_id"] = setlist.get("id")
        result["setlist_url"] = setlist.get("url")

        songs = []
        guests = []

        for set_item in setlist.get("sets", {}).get("set", []):
            for song in set_item.get("song", []):
                song_entry = {
                    "name": song.get("name", "Unknown"),
                    "position": len(songs) + 1,
                    "with_guest": None
                }

                if "with" in song:
                    guest_name = song["with"].get("name")
                    song_entry["with_guest"] = guest_name
                    if guest_name and guest_name not in guests:
                        guests.append(guest_name)

                songs.append(song_entry)

        result["songs"] = songs
        result["song_count"] = len(songs)
        result["guest_artists"] = guests

    return result


def collect_superbowl_halftime_setlists(client: SetlistFMClient) -> Dict[str, Any]:
    """Collect all Super Bowl halftime setlists from 2015-2025."""
    logger.info("Starting Super Bowl halftime setlist collection...")

    halftime_shows = []
    found_count = 0

    for sb_info in SUPERBOWL_HALFTIME_DATA:
        logger.info(f"\n--- Super Bowl {sb_info['super_bowl']} ({sb_info['year']}) ---")

        setlist = None
        for artist in sb_info["artists"]:
            setlist = find_superbowl_setlist(
                client,
                artist_name=artist,
                date=sb_info["date"],
                venue_name=sb_info["venue"],
                city_name=sb_info["city"]
            )
            if setlist:
                break

        halftime_data = extract_halftime_data(setlist, sb_info)
        halftime_shows.append(halftime_data)

        if halftime_data["setlist_found"]:
            found_count += 1
            logger.info(f"  Found {halftime_data['song_count']} songs")

    result = {
        "collection_type": "super_bowl_halftime",
        "collection_timestamp": datetime.utcnow().isoformat(),
        "date_range": "2015-2025",
        "total_shows": len(halftime_shows),
        "setlists_found": found_count,
        "setlists_missing": len(halftime_shows) - found_count,
        "shows": halftime_shows
    }

    logger.info(f"\nCollection complete: {found_count}/{len(halftime_shows)} setlists found")

    return result

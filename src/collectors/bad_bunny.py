"""Collector for Bad Bunny setlist data."""
import logging
from typing import List, Dict, Any
from datetime import datetime

from ..api_client import SetlistFMClient
from ..config import BAD_BUNNY_MBID, PUERTO_RICO_RESIDENCY

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """Parse setlist.fm date format (dd-MM-yyyy) to datetime."""
    return datetime.strptime(date_str, "%d-%m-%Y")


def is_puerto_rico_residency(setlist: Dict[str, Any]) -> bool:
    """Check if a setlist is from the Puerto Rico residency (2024-2025)."""
    venue = setlist.get("venue", {})
    venue_name = venue.get("name", "").lower()
    city = venue.get("city", {})
    city_name = city.get("name", "").lower()
    country = city.get("country", {}).get("name", "").lower()

    is_pr_location = any(
        keyword in venue_name or keyword in city_name
        for keyword in PUERTO_RICO_RESIDENCY["venue_keywords"]
    ) or "puerto rico" in country

    if not is_pr_location:
        return False

    event_date_str = setlist.get("eventDate", "")
    if not event_date_str:
        return False

    try:
        event_date = parse_date(event_date_str)
        start = parse_date(PUERTO_RICO_RESIDENCY["date_range"]["start"])
        end = parse_date(PUERTO_RICO_RESIDENCY["date_range"]["end"])
        return start <= event_date <= end
    except ValueError:
        return False


def extract_setlist_data(setlist: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize setlist data."""
    venue = setlist.get("venue", {})
    city = venue.get("city", {})

    songs = []
    guest_artists = []
    sets_data = setlist.get("sets", {}).get("set", [])

    for set_item in sets_data:
        set_name = set_item.get("name", "Main Set")
        encore = set_item.get("encore")

        for song in set_item.get("song", []):
            song_entry = {
                "name": song.get("name", "Unknown"),
                "set": set_name,
                "encore": encore,
                "is_cover": song.get("cover") is not None,
                "with_guest": None,
                "tape": song.get("tape", False)
            }

            if "with" in song:
                guest_name = song["with"].get("name")
                song_entry["with_guest"] = guest_name
                if guest_name and guest_name not in guest_artists:
                    guest_artists.append(guest_name)

            songs.append(song_entry)

    return {
        "id": setlist.get("id"),
        "event_date": setlist.get("eventDate"),
        "url": setlist.get("url"),
        "venue": {
            "id": venue.get("id"),
            "name": venue.get("name"),
            "city": city.get("name"),
            "state": city.get("state"),
            "country": city.get("country", {}).get("name"),
            "country_code": city.get("country", {}).get("code")
        },
        "tour": setlist.get("tour", {}).get("name") if setlist.get("tour") else None,
        "songs": songs,
        "song_count": len(songs),
        "guest_artists": guest_artists,
        "is_puerto_rico_residency": is_puerto_rico_residency(setlist),
        "priority": "high" if is_puerto_rico_residency(setlist) else "normal",
        "last_updated": setlist.get("lastUpdated")
    }


def collect_bad_bunny_setlists(client: SetlistFMClient) -> Dict[str, Any]:
    """Collect all Bad Bunny setlists from Setlist.fm."""
    logger.info("Starting Bad Bunny setlist collection...")

    # Try MBID first, fall back to name search
    raw_setlists = client.get_all_artist_setlists(BAD_BUNNY_MBID)

    if not raw_setlists:
        logger.info("MBID search returned no results, trying name search...")
        raw_setlists = []
        page = 1
        while True:
            response = client.search_setlists(artist_name="Bad Bunny", page=page)
            setlists = response.get("setlist", [])
            if not setlists:
                break
            raw_setlists.extend(setlists)
            total = response.get("total", 0)
            logger.info(f"Page {page}: found {len(setlists)} setlists (total: {total})")
            if len(raw_setlists) >= total or len(setlists) < 20:
                break
            page += 1

    processed_setlists = []
    pr_residency_count = 0

    for setlist in raw_setlists:
        processed = extract_setlist_data(setlist)
        processed_setlists.append(processed)

        if processed["is_puerto_rico_residency"]:
            pr_residency_count += 1

    processed_setlists.sort(
        key=lambda x: parse_date(x["event_date"]) if x["event_date"] else datetime.min,
        reverse=True
    )

    result = {
        "artist": {
            "name": "Bad Bunny",
            "mbid": BAD_BUNNY_MBID
        },
        "collection_timestamp": datetime.utcnow().isoformat(),
        "total_setlists": len(processed_setlists),
        "puerto_rico_residency_count": pr_residency_count,
        "setlists": processed_setlists
    }

    logger.info(f"Collected {len(processed_setlists)} setlists")
    logger.info(f"Puerto Rico residency shows: {pr_residency_count}")

    return result

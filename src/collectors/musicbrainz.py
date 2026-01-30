"""Collector for Bad Bunny discography from MusicBrainz."""
import json
import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import requests
from requests.exceptions import RequestException

from ..config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

# MusicBrainz API Configuration
MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
USER_AGENT = "BadBunnySetlistPredictor/1.0 (https://github.com/example/badbunnysetlist)"
# Correct MusicBrainz ID for Bad Bunny (found via search)
BAD_BUNNY_MBID = "89aa5ecb-59ad-46f5-b3eb-2d424e941f19"

# Rate limiting: 1 request per second
REQUEST_DELAY = 1.1  # slightly over 1 second to be safe


class MusicBrainzClient:
    """Client for MusicBrainz API with rate limiting."""

    def __init__(self):
        self.base_url = MUSICBRAINZ_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        })
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting (1 request per second)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            sleep_time = REQUEST_DELAY - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Make a GET request to MusicBrainz API."""
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        params["fmt"] = "json"

        for attempt in range(max_retries):
            self._rate_limit()

            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Rate limited or service unavailable
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Not found: {url}")
                    return None
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        return None

            except RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)

        return None

    def get_artist(self, mbid: str) -> Optional[Dict[str, Any]]:
        """Get artist information."""
        return self._make_request(f"/artist/{mbid}")

    def search_artist(self, name: str) -> Optional[Dict[str, Any]]:
        """Search for an artist by name."""
        params = {"query": f'artist:"{name}"', "limit": 5}
        return self._make_request("/artist", params=params)

    def browse_release_groups(
        self,
        artist_mbid: str,
        offset: int = 0,
        limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Browse release groups (albums, singles, EPs) for an artist."""
        params = {
            "artist": artist_mbid,
            "offset": offset,
            "limit": limit
        }
        return self._make_request("/release-group", params=params)

    def get_release_group(self, mbid: str) -> Optional[Dict[str, Any]]:
        """Get release group with releases."""
        return self._make_request(
            f"/release-group/{mbid}",
            params={"inc": "releases+artist-credits"}
        )

    def get_release(self, mbid: str) -> Optional[Dict[str, Any]]:
        """Get release with recordings."""
        return self._make_request(
            f"/release/{mbid}",
            params={"inc": "recordings+artist-credits+media"}
        )

    def browse_recordings(
        self,
        artist_mbid: str,
        offset: int = 0,
        limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Browse all recordings for an artist."""
        params = {
            "artist": artist_mbid,
            "offset": offset,
            "limit": limit,
            "inc": "artist-credits"  # Only valid inc params for recording browse
        }
        return self._make_request("/recording", params=params)


def get_all_release_groups(client: MusicBrainzClient, artist_mbid: str) -> List[Dict[str, Any]]:
    """Get all release groups for an artist with pagination."""
    all_groups = []
    offset = 0
    limit = 100

    while True:
        logger.info(f"Fetching release groups offset={offset}...")
        response = client.browse_release_groups(artist_mbid, offset=offset, limit=limit)

        if not response:
            break

        groups = response.get("release-groups", [])
        if not groups:
            break

        all_groups.extend(groups)
        total = response.get("release-group-count", 0)
        logger.info(f"Got {len(groups)} release groups (total: {total})")

        if len(all_groups) >= total:
            break

        offset += limit

    return all_groups


def get_all_recordings(client: MusicBrainzClient, artist_mbid: str) -> List[Dict[str, Any]]:
    """Get all recordings for an artist with pagination."""
    all_recordings = []
    offset = 0
    limit = 100

    while True:
        logger.info(f"Fetching recordings offset={offset}...")
        response = client.browse_recordings(artist_mbid, offset=offset, limit=limit)

        if not response:
            break

        recordings = response.get("recordings", [])
        if not recordings:
            break

        all_recordings.extend(recordings)
        total = response.get("recording-count", 0)
        logger.info(f"Got {len(recordings)} recordings (total: {total})")

        if len(all_recordings) >= total:
            break

        offset += limit

    return all_recordings


def is_bad_bunny_primary(artist_credits: List[Dict[str, Any]], artist_mbid: str) -> bool:
    """Check if Bad Bunny is the primary artist."""
    if not artist_credits:
        return False
    # Primary artist is usually first in credits
    first_artist = artist_credits[0].get("artist", {})
    return first_artist.get("id") == artist_mbid


def get_all_artists(artist_credits: List[Dict[str, Any]]) -> List[str]:
    """Extract all artist names from credits."""
    artists = []
    for credit in artist_credits:
        artist_name = credit.get("artist", {}).get("name")
        if artist_name:
            artists.append(artist_name)
    return artists


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse various date formats from MusicBrainz."""
    if not date_str:
        return None
    try:
        # Try full date
        if len(date_str) == 10:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        # Try year-month
        elif len(date_str) == 7:
            return datetime.strptime(date_str, "%Y-%m").date()
        # Try year only
        elif len(date_str) == 4:
            return datetime.strptime(date_str, "%Y").date()
    except ValueError:
        pass
    return None


def calculate_song_age_days(release_date: Optional[date]) -> Optional[int]:
    """Calculate song age in days from release date."""
    if not release_date:
        return None
    today = date.today()
    return (today - release_date).days


def load_setlist_data() -> Dict[str, int]:
    """Load setlist data and count song performances."""
    setlist_file = DATA_RAW_DIR / "bad_bunny_setlists.json"
    song_counts: Dict[str, int] = {}

    if not setlist_file.exists():
        logger.warning("Setlist data not found, skipping performance counts")
        return song_counts

    with open(setlist_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for setlist in data.get("setlists", []):
        for song in setlist.get("songs", []):
            song_name = song.get("name", "").lower().strip()
            if song_name:
                song_counts[song_name] = song_counts.get(song_name, 0) + 1

    logger.info(f"Loaded performance counts for {len(song_counts)} songs")
    return song_counts


def normalize_song_name(name: str) -> str:
    """Normalize song name for matching."""
    import re
    # Lowercase, remove special characters, normalize whitespace
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def find_artist_mbid(client: MusicBrainzClient, name: str) -> Optional[str]:
    """Search for artist and return MBID."""
    response = client.search_artist(name)
    if response and response.get("artists"):
        for artist in response["artists"]:
            if artist.get("name", "").lower() == name.lower():
                return artist.get("id")
        # If exact match not found, return first result
        return response["artists"][0].get("id")
    return None


def collect_bad_bunny_catalog(client: MusicBrainzClient) -> Dict[str, Any]:
    """Collect Bad Bunny's complete catalog from MusicBrainz."""
    logger.info("Starting Bad Bunny catalog collection from MusicBrainz...")

    # Try configured MBID first, fall back to search
    artist_mbid = BAD_BUNNY_MBID

    # Verify MBID works by getting artist info
    artist_info = client.get_artist(artist_mbid)
    if not artist_info:
        logger.info("Configured MBID not found, searching by name...")
        artist_mbid = find_artist_mbid(client, "Bad Bunny")
        if not artist_mbid:
            logger.error("Could not find Bad Bunny on MusicBrainz!")
            return {"error": "Artist not found"}
        logger.info(f"Found MBID via search: {artist_mbid}")

    logger.info(f"Using MBID: {artist_mbid}")

    # Load setlist data for performance counts
    setlist_counts = load_setlist_data()

    # Get all recordings for the artist
    recordings = get_all_recordings(client, artist_mbid)
    logger.info(f"Found {len(recordings)} total recordings")

    # Get release groups for album info
    release_groups = get_all_release_groups(client, artist_mbid)
    logger.info(f"Found {len(release_groups)} total release groups")

    # Build release group lookup
    release_group_info: Dict[str, Dict[str, Any]] = {}
    for rg in release_groups:
        rg_id = rg.get("id")
        if rg_id:
            release_group_info[rg_id] = {
                "title": rg.get("title"),
                "first_release_date": rg.get("first-release-date"),
                "type": rg.get("primary-type"),
                "secondary_types": rg.get("secondary-types", [])
            }

    # Process recordings into catalog
    tracks: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()  # Deduplicate by normalized title

    # Get latest album date for recency boost calculation
    latest_album_date = None
    for rg in release_groups:
        if rg.get("primary-type") == "Album":
            rg_date = parse_date(rg.get("first-release-date"))
            if rg_date and (latest_album_date is None or rg_date > latest_album_date):
                latest_album_date = rg_date

    logger.info(f"Latest album date: {latest_album_date}")

    for recording in recordings:
        track_name = recording.get("title", "")
        normalized_name = normalize_song_name(track_name)

        # Skip if we've already seen this song (dedup)
        if normalized_name in seen_titles:
            continue
        seen_titles.add(normalized_name)

        # Get artist credits
        artist_credits = recording.get("artist-credit", [])
        is_primary = is_bad_bunny_primary(artist_credits, artist_mbid)
        all_artists = get_all_artists(artist_credits)

        # We don't have release info from recording browse, so we'll
        # try to match with release groups by looking at the recording's releases later
        # For now, just use None for release info
        release_date_str = None
        release_name = None
        release_type = None

        # Try to find a matching release group by title similarity
        # (This is a simplified approach - in production you'd want to fetch release details)
        release_date = None

        # Calculate metrics
        song_age_days = calculate_song_age_days(release_date)

        # Look up performance count (try multiple name variations)
        times_performed = 0
        for variation in [normalized_name, track_name.lower().strip()]:
            if variation in setlist_counts:
                times_performed = setlist_counts[variation]
                break

        # Recency boost: songs from the last 2 years get a boost
        recency_boost = False
        if release_date and latest_album_date:
            days_since_latest = (latest_album_date - release_date).days
            recency_boost = days_since_latest <= 730  # Within 2 years of latest album

        track_entry = {
            "track_name": track_name,
            "track_mbid": recording.get("id"),
            "duration_seconds": (recording.get("length") or 0) // 1000,  # ms to seconds
            "release_name": release_name,
            "release_date": release_date_str,
            "release_type": release_type,
            "is_primary_artist": is_primary,
            "all_artists": all_artists,
            "times_performed_live": times_performed,
            "song_age_days": song_age_days,
            "recency_boost": recency_boost
        }
        tracks.append(track_entry)

    # Sort by performance count (popularity proxy)
    tracks.sort(key=lambda x: x["times_performed_live"], reverse=True)

    # Compile album list from release groups
    albums = []
    seen_album_titles: Set[str] = set()
    for rg in release_groups:
        title = rg.get("title", "")
        rg_type = rg.get("primary-type")
        if title not in seen_album_titles and rg_type in ["Album", "EP", "Single"]:
            seen_album_titles.add(title)
            albums.append({
                "title": title,
                "release_date": rg.get("first-release-date"),
                "type": rg_type,
                "mbid": rg.get("id")
            })

    # Sort albums by date
    albums.sort(key=lambda x: x.get("release_date") or "", reverse=True)

    # Summary stats
    primary_tracks = sum(1 for t in tracks if t["is_primary_artist"])
    featured_tracks = sum(1 for t in tracks if not t["is_primary_artist"])
    performed_tracks = sum(1 for t in tracks if t["times_performed_live"] > 0)

    result = {
        "artist": {
            "name": "Bad Bunny",
            "mbid": artist_mbid
        },
        "collection_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tracks": len(tracks),
            "primary_artist_tracks": primary_tracks,
            "featured_tracks": featured_tracks,
            "tracks_performed_live": performed_tracks,
            "total_albums_eps": len(albums),
            "latest_album_date": str(latest_album_date) if latest_album_date else None
        },
        "albums": albums,
        "tracks": tracks
    }

    logger.info(f"Catalog complete: {len(tracks)} tracks, {len(albums)} albums/EPs")
    logger.info(f"Primary: {primary_tracks}, Featured: {featured_tracks}")
    logger.info(f"Tracks performed live: {performed_tracks}")

    return result


def main():
    """Main function to collect Bad Bunny catalog."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 60)
    logger.info("Bad Bunny Catalog Collection from MusicBrainz")
    logger.info("=" * 60)

    # Ensure output directory exists
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize client and collect
    client = MusicBrainzClient()
    catalog = collect_bad_bunny_catalog(client)

    # Save to file
    output_path = DATA_RAW_DIR / "bad_bunny_catalog.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved catalog to: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CATALOG SUMMARY")
    logger.info("=" * 60)
    summary = catalog["summary"]
    logger.info(f"Total tracks: {summary['total_tracks']}")
    logger.info(f"Primary artist: {summary['primary_artist_tracks']}")
    logger.info(f"Featured: {summary['featured_tracks']}")
    logger.info(f"Performed live: {summary['tracks_performed_live']}")
    logger.info(f"Albums/EPs: {summary['total_albums_eps']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

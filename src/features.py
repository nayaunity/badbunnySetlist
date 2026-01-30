"""Feature engineering for Bad Bunny setlist prediction model."""
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd

from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
TODAY = date(2026, 1, 30)
LATEST_ALBUM = "Debí Tirar Más Fotos"
LATEST_ALBUM_NORMALIZED = "debi tirar mas fotos"


def normalize_song_name(name: str) -> str:
    """Normalize song name for matching."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def load_catalog() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the song catalog and release groups."""
    catalog_path = DATA_RAW_DIR / "bad_bunny_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tracks", []), data.get("albums", [])


def load_setlists() -> Dict[str, Any]:
    """Load setlist data."""
    setlist_path = DATA_RAW_DIR / "bad_bunny_setlists.json"
    with open(setlist_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_setlist_date(date_str: str) -> Optional[date]:
    """Parse setlist date format (dd-MM-yyyy)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").date()
    except ValueError:
        return None


def analyze_setlists(setlists_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, int, int]:
    """
    Analyze setlists to extract performance statistics.

    Returns:
        - song_positions: dict of song_name -> list of positions (1-indexed)
        - song_counts: dict of song_name -> total performance count
        - pr_residency_counts: dict of song_name -> count in PR residency shows
        - first_seen_dates: dict of song_name -> earliest setlist date
        - total_shows: total number of shows
        - pr_shows: number of PR residency shows
    """
    song_positions: Dict[str, List[int]] = defaultdict(list)
    song_counts: Dict[str, int] = defaultdict(int)
    pr_residency_counts: Dict[str, int] = defaultdict(int)
    first_seen_dates: Dict[str, date] = {}

    setlists = setlists_data.get("setlists", [])
    total_shows = len(setlists)
    pr_shows = 0

    for setlist in setlists:
        is_pr = setlist.get("is_puerto_rico_residency", False)
        if is_pr:
            pr_shows += 1

        setlist_date = parse_setlist_date(setlist.get("event_date", ""))
        songs = setlist.get("songs", [])
        total_songs = len(songs)

        for position, song in enumerate(songs, 1):
            song_name = normalize_song_name(song.get("name", ""))
            if not song_name:
                continue

            # Track first seen date
            if setlist_date:
                if song_name not in first_seen_dates or setlist_date < first_seen_dates[song_name]:
                    first_seen_dates[song_name] = setlist_date

            song_counts[song_name] += 1

            # Normalize position to 0-1 scale (0 = opener, 1 = closer)
            if total_songs > 1:
                normalized_pos = (position - 1) / (total_songs - 1)
            else:
                normalized_pos = 0.5
            song_positions[song_name].append(normalized_pos)

            if is_pr:
                pr_residency_counts[song_name] += 1

    logger.info(f"Analyzed {total_shows} shows ({pr_shows} PR residency)")
    logger.info(f"Found {len(song_counts)} unique songs performed")
    logger.info(f"First seen dates tracked for {len(first_seen_dates)} songs")

    return song_positions, song_counts, pr_residency_counts, first_seen_dates, total_shows, pr_shows


def parse_release_date(date_str: Optional[str]) -> Optional[date]:
    """Parse release date string to date object."""
    if not date_str:
        return None
    try:
        if len(date_str) == 10:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        elif len(date_str) == 7:
            return datetime.strptime(date_str + "-01", "%Y-%m-%d").date()
        elif len(date_str) == 4:
            return datetime.strptime(date_str + "-01-01", "%Y-%m-%d").date()
    except ValueError:
        pass
    return None


def calculate_setlist_position_category(avg_position: float) -> str:
    """Categorize setlist position into opener/middle/closer."""
    if avg_position <= 0.2:
        return "opener"
    elif avg_position >= 0.8:
        return "closer"
    else:
        return "middle"


def build_release_group_lookup(albums: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup of release group info by normalized title."""
    lookup = {}
    for album in albums:
        title = album.get("title", "")
        normalized = normalize_song_name(title)
        if normalized and normalized not in lookup:
            lookup[normalized] = {
                "title": title,
                "release_date": album.get("release_date"),
                "type": album.get("type")
            }
    return lookup


def build_feature_matrix(
    catalog: List[Dict[str, Any]],
    albums: List[Dict[str, Any]],
    song_positions: Dict[str, List[float]],
    song_counts: Dict[str, int],
    pr_residency_counts: Dict[str, int],
    first_seen_dates: Dict[str, date],
    total_shows: int,
    pr_shows: int
) -> pd.DataFrame:
    """Build the feature matrix from catalog and setlist analysis."""

    # Build release group lookup for date inference
    release_lookup = build_release_group_lookup(albums)

    rows = []

    for track in catalog:
        track_name = track.get("track_name", "")
        normalized_name = normalize_song_name(track_name)

        # === Identifiers (not features) ===
        song_id = track.get("track_mbid", "")

        # === 1. Live Performance Features ===
        times_performed = song_counts.get(normalized_name, 0)
        performance_frequency = times_performed / total_shows if total_shows > 0 else 0
        ever_performed_live = 1 if times_performed > 0 else 0

        # Average setlist position
        positions = song_positions.get(normalized_name, [])
        if positions:
            avg_position = sum(positions) / len(positions)
            position_category = calculate_setlist_position_category(avg_position)
        else:
            avg_position = None
            position_category = None

        # === 2. Recency Features ===
        # Try catalog release date first, then release group lookup
        release_date_str = track.get("release_date")
        release_name = track.get("release_name", "") or ""
        release_date = parse_release_date(release_date_str)

        # If no release date, try to get from release group by matching track name
        # to album title (for title tracks) or use first seen date as proxy
        if not release_date and release_name:
            rg_info = release_lookup.get(normalize_song_name(release_name), {})
            release_date = parse_release_date(rg_info.get("release_date"))

        # Use first seen in setlist as proxy for release date if still missing
        first_seen = first_seen_dates.get(normalized_name)

        if release_date:
            days_since_release = (TODAY - release_date).days
            release_year = release_date.year
        elif first_seen:
            # Use first seen as approximation (song released before first performance)
            days_since_release = (TODAY - first_seen).days
            release_year = first_seen.year
        else:
            days_since_release = None
            release_year = None

        # Check if from latest album "Debí Tirar Más Fotos"
        is_latest_album = 1 if LATEST_ALBUM_NORMALIZED in normalize_song_name(release_name) else 0

        # Also check track name against known latest album tracks
        latest_album_tracks = [
            "nuevayol", "dtmf", "vou 787", "baile inolvidable", "perfumito nuevo",
            "weltita", "ketu tecre", "pitorro de coco", "bokete", "el club",
            "turista", "lo que le paso a hawaii", "cafe con ron", "veldá"
        ]
        if normalized_name in latest_album_tracks:
            is_latest_album = 1

        # === 3. Artist Features ===
        is_primary = 1 if track.get("is_primary_artist", False) else 0
        all_artists = track.get("all_artists", [])
        num_featured_artists = max(0, len(all_artists) - 1)  # Subtract Bad Bunny
        has_featured_artist = 1 if num_featured_artists > 0 else 0

        # === 4. Duration Features ===
        duration_seconds = track.get("duration_seconds", 0) or 0

        if duration_seconds < 180:
            duration_bucket = "short"
        elif duration_seconds <= 240:
            duration_bucket = "medium"
        else:
            duration_bucket = "long"

        # === 5. Puerto Rico Residency Signal ===
        pr_count = pr_residency_counts.get(normalized_name, 0)
        performed_at_pr_residency = 1 if pr_count > 0 else 0
        pr_residency_frequency = pr_count / pr_shows if pr_shows > 0 else 0

        # Build row
        row = {
            # Identifiers
            "song_name": track_name,
            "song_id": song_id,

            # 1. Live Performance Features
            "times_performed_live": times_performed,
            "performance_frequency": round(performance_frequency, 4),
            "ever_performed_live": ever_performed_live,
            "avg_setlist_position": round(avg_position, 4) if avg_position is not None else None,
            "position_category": position_category,

            # 2. Recency Features
            "days_since_release": days_since_release,
            "is_latest_album": is_latest_album,
            "release_year": release_year,

            # 3. Artist Features
            "is_primary_artist": is_primary,
            "num_featured_artists": num_featured_artists,
            "has_featured_artist": has_featured_artist,

            # 4. Duration Features
            "duration_seconds": duration_seconds,
            "duration_bucket": duration_bucket,

            # 5. Puerto Rico Residency Signal
            "performed_at_pr_residency": performed_at_pr_residency,
            "pr_residency_frequency": round(pr_residency_frequency, 4)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the feature matrix."""

    print("\n" + "=" * 70)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 70)

    print(f"\nTotal songs: {len(df)}")
    print(f"Songs performed live: {df['ever_performed_live'].sum()}")
    print(f"Songs never performed: {len(df) - df['ever_performed_live'].sum()}")

    # === Missing Values ===
    print("\n" + "-" * 40)
    print("MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0]
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("No missing values!")

    # === Numeric Features Distribution ===
    print("\n" + "-" * 40)
    print("NUMERIC FEATURES DISTRIBUTION")
    print("-" * 40)
    numeric_cols = [
        "times_performed_live", "performance_frequency", "avg_setlist_position",
        "days_since_release", "duration_seconds", "num_featured_artists",
        "pr_residency_frequency"
    ]
    print(df[numeric_cols].describe().round(2).to_string())

    # === Binary Features ===
    print("\n" + "-" * 40)
    print("BINARY FEATURES")
    print("-" * 40)
    binary_cols = [
        "ever_performed_live", "is_latest_album", "is_primary_artist",
        "has_featured_artist", "performed_at_pr_residency"
    ]
    for col in binary_cols:
        pct = df[col].mean() * 100
        print(f"{col}: {df[col].sum()} / {len(df)} ({pct:.1f}%)")

    # === Categorical Features ===
    print("\n" + "-" * 40)
    print("CATEGORICAL FEATURES")
    print("-" * 40)

    print("\nPosition Category (for performed songs):")
    pos_counts = df[df["position_category"].notna()]["position_category"].value_counts()
    for cat, count in pos_counts.items():
        print(f"  {cat}: {count}")

    print("\nDuration Bucket:")
    dur_counts = df["duration_bucket"].value_counts()
    for bucket, count in dur_counts.items():
        print(f"  {bucket}: {count}")

    print("\nRelease Year Distribution:")
    year_counts = df["release_year"].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count}")

    # === Top Performed Songs ===
    print("\n" + "-" * 40)
    print("TOP 10 MOST PERFORMED SONGS")
    print("-" * 40)
    top_performed = df.nlargest(10, "times_performed_live")[
        ["song_name", "times_performed_live", "performance_frequency", "is_latest_album"]
    ]
    print(top_performed.to_string(index=False))

    # === Latest Album Songs ===
    print("\n" + "-" * 40)
    print("LATEST ALBUM SONGS PERFORMANCE")
    print("-" * 40)
    latest_album = df[df["is_latest_album"] == 1].sort_values(
        "times_performed_live", ascending=False
    )[["song_name", "times_performed_live", "performed_at_pr_residency"]]
    if len(latest_album) > 0:
        print(latest_album.head(15).to_string(index=False))
    else:
        print("No songs from latest album found in catalog")

    print("\n" + "=" * 70)


def main():
    """Main function to build feature matrix."""
    logger.info("=" * 60)
    logger.info("Building Feature Matrix for Bad Bunny Setlist Prediction")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading catalog...")
    catalog, albums = load_catalog()
    logger.info(f"Loaded {len(catalog)} tracks and {len(albums)} albums from catalog")

    logger.info("Loading setlists...")
    setlists_data = load_setlists()

    # Analyze setlists
    logger.info("Analyzing setlist data...")
    song_positions, song_counts, pr_counts, first_seen, total_shows, pr_shows = analyze_setlists(setlists_data)

    # Build feature matrix
    logger.info("Building feature matrix...")
    df = build_feature_matrix(
        catalog, albums, song_positions, song_counts, pr_counts, first_seen, total_shows, pr_shows
    )

    # Ensure output directory exists
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = DATA_PROCESSED_DIR / "feature_matrix.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved feature matrix to: {output_path}")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Print summary
    print_feature_summary(df)


if __name__ == "__main__":
    main()

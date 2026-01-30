"""Constraint-based setlist optimization for Super Bowl prediction."""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from .config import DATA_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# === CONSTRAINTS CONFIGURATION ===

class SetlistConstraints:
    """Configuration for setlist constraints."""

    # Time constraints (Super Bowl halftimes are typically 12-14 minutes)
    TARGET_DURATION_SECONDS = 780  # 13 minutes
    MIN_DURATION_SECONDS = 720     # 12 minutes
    MAX_DURATION_SECONDS = 840     # 14 minutes

    # Song count constraints
    MIN_SONGS = 8
    MAX_SONGS = 12

    # Medley duration multiplier (songs shortened for medley)
    SNIPPET_DURATION_RATIO = 0.45  # 45% of original for tight Super Bowl medleys

    # Album representation
    MIN_LATEST_ALBUM_SONGS = 2
    MAX_LATEST_ALBUM_SONGS = 4
    MIN_CLASSIC_HITS = 3  # Songs with high performance count
    CLASSIC_HIT_THRESHOLD = 50  # Minimum performances to be "classic"

    # Collaboration constraints
    MAX_COLLAB_SONGS = 3

    # Energy positions (normalized 0-1)
    OPENER_THRESHOLD = 0.15   # First ~15% of setlist
    CLOSER_THRESHOLD = 0.85   # Last ~15% of setlist


class SongFormat(Enum):
    """Song performance format."""
    FULL = "full"
    SNIPPET = "snippet"


@dataclass
class SetlistEntry:
    """A song entry in the predicted setlist."""
    position: int
    song_name: str
    song_id: str
    format: str  # "full" or "snippet"
    estimated_duration: int  # seconds
    original_duration: int
    combined_score: float
    is_latest_album: bool
    is_classic_hit: bool
    has_featured_artist: bool
    is_confirmed: bool
    placement_type: str  # "opener", "middle", "closer"
    selection_reason: str


@dataclass
class PredictedSetlist:
    """The complete predicted setlist."""
    generated_at: str
    total_songs: int
    total_duration_seconds: int
    total_duration_formatted: str
    confidence_score: float
    constraints_satisfied: Dict[str, bool]
    setlist: List[Dict[str, Any]]
    alternates: List[Dict[str, Any]]
    methodology: str


def load_training_data() -> pd.DataFrame:
    """Load the training data with all features and labels."""
    path = DATA_PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def get_confirmed_songs(df: pd.DataFrame) -> List[str]:
    """Get songs that are confirmed for the halftime show."""
    confirmed = df[
        (df["in_halftime_trailer"] == 1) |
        (df["confirmed_for_halftime"] == 1)
    ]["song_name"].tolist()
    return confirmed


def classify_song(row: pd.Series) -> Dict[str, bool]:
    """Classify a song's characteristics."""
    return {
        "is_latest_album": row.get("is_latest_album", 0) == 1,
        "is_classic_hit": row.get("times_performed_live", 0) >= SetlistConstraints.CLASSIC_HIT_THRESHOLD,
        "has_featured_artist": row.get("has_featured_artist", 0) == 1,
        "is_major_collab": row.get("major_collab_artist", 0) == 1,
        "is_confirmed": (row.get("in_halftime_trailer", 0) == 1 or
                        row.get("confirmed_for_halftime", 0) == 1),
        "high_cultural": row.get("cultural_significance", 0) >= 2,
    }


def estimate_duration(original_duration: int, format: SongFormat) -> int:
    """Estimate performance duration based on format."""
    if original_duration <= 0:
        # Default duration if unknown (typical song ~3:20)
        original_duration = 200

    if format == SongFormat.SNIPPET:
        return int(original_duration * SetlistConstraints.SNIPPET_DURATION_RATIO)
    return original_duration


def get_song_duration(row) -> int:
    """Get song duration with fallback for missing data."""
    duration = row.get("duration_seconds", 0)
    if pd.isna(duration) or duration <= 0:
        # Use typical song duration as fallback
        return 180  # ~3:00
    return int(duration)


def get_placement_type(position: int, total_songs: int) -> str:
    """Determine placement type based on position."""
    if total_songs <= 0:
        return "middle"

    normalized_pos = position / total_songs

    if normalized_pos <= SetlistConstraints.OPENER_THRESHOLD:
        return "opener"
    elif normalized_pos >= SetlistConstraints.CLOSER_THRESHOLD:
        return "closer"
    return "middle"


def calculate_energy_score(row: pd.Series, placement_type: str) -> float:
    """
    Calculate how well a song fits a placement based on energy/performance patterns.
    Uses avg_setlist_position from historical data.
    """
    avg_pos = row.get("avg_setlist_position")

    if pd.isna(avg_pos):
        # No position data - neutral score
        return 0.5

    # Songs typically played as openers (low avg_pos) fit opener slots
    # Songs typically played as closers (high avg_pos) fit closer slots
    if placement_type == "opener":
        # Prefer songs with low avg_setlist_position
        return 1.0 - avg_pos
    elif placement_type == "closer":
        # Prefer songs with high avg_setlist_position
        return avg_pos
    else:
        # Middle songs - prefer those typically in middle
        return 1.0 - abs(avg_pos - 0.5) * 2


def select_songs_greedy(
    df: pd.DataFrame,
    confirmed_songs: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Greedy selection algorithm with constraint satisfaction.

    Returns:
        - selected: list of selected song entries
        - alternates: list of alternate candidates
    """
    selected = []
    alternates = []

    # Track constraint satisfaction
    total_duration = 0
    latest_album_count = 0
    classic_hit_count = 0
    collab_count = 0

    # Sort by combined_score descending
    df_sorted = df.sort_values("combined_score", ascending=False).copy()

    # Phase 1: Add confirmed songs first
    for song_name in confirmed_songs:
        song_row = df_sorted[df_sorted["song_name"] == song_name]
        if len(song_row) == 0:
            continue

        row = song_row.iloc[0]
        classification = classify_song(row)

        # Determine format based on duration
        original_dur = get_song_duration(row)

        # Confirmed songs get more time - full or long snippet
        if original_dur > 240:
            fmt = SongFormat.SNIPPET
            duration = estimate_duration(original_dur, SongFormat.SNIPPET)
        else:
            fmt = SongFormat.FULL
            duration = original_dur

        entry = {
            "song_name": song_name,
            "song_id": row.get("song_id", ""),
            "format": fmt.value,
            "estimated_duration": duration,
            "original_duration": original_dur,
            "combined_score": float(row.get("combined_score", 0)),
            "is_latest_album": classification["is_latest_album"],
            "is_classic_hit": classification["is_classic_hit"],
            "has_featured_artist": classification["has_featured_artist"],
            "is_confirmed": True,
            "selection_reason": "Confirmed in halftime trailer"
        }

        selected.append(entry)
        total_duration += duration

        if classification["is_latest_album"]:
            latest_album_count += 1
        if classification["is_classic_hit"]:
            classic_hit_count += 1
        if classification["has_featured_artist"]:
            collab_count += 1

        # Remove from candidates
        df_sorted = df_sorted[df_sorted["song_name"] != song_name]

    logger.info(f"Phase 1: Added {len(selected)} confirmed songs ({total_duration}s)")

    # Phase 2: Ensure latest album representation
    if latest_album_count < SetlistConstraints.MIN_LATEST_ALBUM_SONGS:
        needed = SetlistConstraints.MIN_LATEST_ALBUM_SONGS - latest_album_count
        latest_album_candidates = df_sorted[df_sorted["is_latest_album"] == 1].head(needed + 2)

        for _, row in latest_album_candidates.iterrows():
            if latest_album_count >= SetlistConstraints.MIN_LATEST_ALBUM_SONGS:
                break
            if len(selected) >= SetlistConstraints.MAX_SONGS:
                break

            classification = classify_song(row)
            original_dur = get_song_duration(row)

            # Latest album songs - give them decent time to showcase
            if original_dur > 200:
                fmt = SongFormat.SNIPPET
                duration = estimate_duration(original_dur, SongFormat.SNIPPET)
            else:
                fmt = SongFormat.FULL
                duration = original_dur

            # Check time constraint
            if total_duration + duration > SetlistConstraints.MAX_DURATION_SECONDS:
                continue

            entry = {
                "song_name": row["song_name"],
                "song_id": row.get("song_id", ""),
                "format": fmt.value,
                "estimated_duration": duration,
                "original_duration": original_dur,
                "combined_score": float(row.get("combined_score", 0)),
                "is_latest_album": True,
                "is_classic_hit": classification["is_classic_hit"],
                "has_featured_artist": classification["has_featured_artist"],
                "is_confirmed": False,
                "selection_reason": "Latest album representation"
            }

            selected.append(entry)
            total_duration += duration
            latest_album_count += 1

            if classification["is_classic_hit"]:
                classic_hit_count += 1
            if classification["has_featured_artist"]:
                collab_count += 1

            df_sorted = df_sorted[df_sorted["song_name"] != row["song_name"]]

    logger.info(f"Phase 2: Latest album songs = {latest_album_count}")

    # Phase 3: Ensure classic hits
    if classic_hit_count < SetlistConstraints.MIN_CLASSIC_HITS:
        needed = SetlistConstraints.MIN_CLASSIC_HITS - classic_hit_count
        classic_candidates = df_sorted[
            df_sorted["times_performed_live"] >= SetlistConstraints.CLASSIC_HIT_THRESHOLD
        ].head(needed + 3)

        for _, row in classic_candidates.iterrows():
            if classic_hit_count >= SetlistConstraints.MIN_CLASSIC_HITS:
                break
            if len(selected) >= SetlistConstraints.MAX_SONGS:
                break

            classification = classify_song(row)

            # Skip if too many collabs
            if classification["has_featured_artist"] and collab_count >= SetlistConstraints.MAX_COLLAB_SONGS:
                continue

            original_dur = get_song_duration(row)

            # Classic hits - can be snippets to fit more
            fmt = SongFormat.SNIPPET
            duration = estimate_duration(original_dur, SongFormat.SNIPPET)

            if total_duration + duration > SetlistConstraints.MAX_DURATION_SECONDS:
                continue

            entry = {
                "song_name": row["song_name"],
                "song_id": row.get("song_id", ""),
                "format": fmt.value,
                "estimated_duration": duration,
                "original_duration": original_dur,
                "combined_score": float(row.get("combined_score", 0)),
                "is_latest_album": classification["is_latest_album"],
                "is_classic_hit": True,
                "has_featured_artist": classification["has_featured_artist"],
                "is_confirmed": False,
                "selection_reason": "Classic hit inclusion"
            }

            selected.append(entry)
            total_duration += duration
            classic_hit_count += 1

            if classification["has_featured_artist"]:
                collab_count += 1

            df_sorted = df_sorted[df_sorted["song_name"] != row["song_name"]]

    logger.info(f"Phase 3: Classic hits = {classic_hit_count}")

    # Phase 4: Fill remaining slots with highest combined_score
    while len(selected) < SetlistConstraints.MIN_SONGS:
        if len(df_sorted) == 0:
            break

        added_song = False
        for _, row in df_sorted.iterrows():
            if len(selected) >= SetlistConstraints.MAX_SONGS:
                break

            classification = classify_song(row)

            # Skip if too many collabs
            if classification["has_featured_artist"] and collab_count >= SetlistConstraints.MAX_COLLAB_SONGS:
                continue

            # Skip if too many latest album songs
            if classification["is_latest_album"] and latest_album_count >= SetlistConstraints.MAX_LATEST_ALBUM_SONGS:
                continue

            original_dur = get_song_duration(row)

            # Determine format based on remaining time
            remaining_time = SetlistConstraints.TARGET_DURATION_SECONDS - total_duration
            remaining_songs = SetlistConstraints.MIN_SONGS - len(selected)
            avg_time_per_song = remaining_time / max(1, remaining_songs)

            if original_dur > avg_time_per_song * 1.2:
                fmt = SongFormat.SNIPPET
                duration = estimate_duration(original_dur, SongFormat.SNIPPET)
            else:
                fmt = SongFormat.FULL
                duration = original_dur

            if total_duration + duration > SetlistConstraints.MAX_DURATION_SECONDS:
                # Try snippet instead
                fmt = SongFormat.SNIPPET
                duration = estimate_duration(original_dur, SongFormat.SNIPPET)
                if total_duration + duration > SetlistConstraints.MAX_DURATION_SECONDS:
                    continue

            entry = {
                "song_name": row["song_name"],
                "song_id": row.get("song_id", ""),
                "format": fmt.value,
                "estimated_duration": duration,
                "original_duration": original_dur,
                "combined_score": float(row.get("combined_score", 0)),
                "is_latest_album": classification["is_latest_album"],
                "is_classic_hit": classification["is_classic_hit"],
                "has_featured_artist": classification["has_featured_artist"],
                "is_confirmed": False,
                "selection_reason": "High combined score"
            }

            selected.append(entry)
            total_duration += duration

            if classification["is_latest_album"]:
                latest_album_count += 1
            if classification["is_classic_hit"]:
                classic_hit_count += 1
            if classification["has_featured_artist"]:
                collab_count += 1

            df_sorted = df_sorted[df_sorted["song_name"] != row["song_name"]]
            added_song = True
            break

        # Break if no song could be added (all remaining songs filtered out)
        if not added_song:
            break

    logger.info(f"Phase 4: Total songs = {len(selected)}, Duration = {total_duration}s")

    # Phase 5: Try to add more songs if under min and time allows
    while len(selected) < SetlistConstraints.MAX_SONGS and total_duration < SetlistConstraints.TARGET_DURATION_SECONDS - 60:
        added = False
        for _, row in df_sorted.iterrows():
            classification = classify_song(row)

            if classification["has_featured_artist"] and collab_count >= SetlistConstraints.MAX_COLLAB_SONGS:
                continue

            original_dur = get_song_duration(row)
            fmt = SongFormat.SNIPPET
            duration = estimate_duration(original_dur, SongFormat.SNIPPET)

            if total_duration + duration > SetlistConstraints.MAX_DURATION_SECONDS:
                continue

            entry = {
                "song_name": row["song_name"],
                "song_id": row.get("song_id", ""),
                "format": fmt.value,
                "estimated_duration": duration,
                "original_duration": original_dur,
                "combined_score": float(row.get("combined_score", 0)),
                "is_latest_album": classification["is_latest_album"],
                "is_classic_hit": classification["is_classic_hit"],
                "has_featured_artist": classification["has_featured_artist"],
                "is_confirmed": False,
                "selection_reason": "Additional high-score song"
            }

            selected.append(entry)
            total_duration += duration

            if classification["has_featured_artist"]:
                collab_count += 1

            df_sorted = df_sorted[df_sorted["song_name"] != row["song_name"]]
            added = True
            break

        if not added:
            break

    # Collect alternates (next best songs not selected)
    for _, row in df_sorted.head(10).iterrows():
        classification = classify_song(row)
        alternates.append({
            "song_name": row["song_name"],
            "combined_score": float(row.get("combined_score", 0)),
            "is_latest_album": classification["is_latest_album"],
            "is_classic_hit": classification["is_classic_hit"],
            "reason_not_selected": "Constraint limits reached"
        })

    return selected, alternates


def order_setlist_by_energy(selected: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Order the selected songs to create an energy arc.

    Strategy:
    - Strong opener (high energy, crowd-pleaser)
    - Build through middle
    - Peak energy finale
    """
    if len(selected) == 0:
        return selected

    # Create lookup for avg_setlist_position
    pos_lookup = dict(zip(df["song_name"], df["avg_setlist_position"]))

    # Separate confirmed songs (they might have specific placement needs)
    confirmed = [s for s in selected if s.get("is_confirmed", False)]
    others = [s for s in selected if not s.get("is_confirmed", False)]

    # Score songs for opener/closer suitability
    def opener_score(song):
        name = song["song_name"]
        avg_pos = pos_lookup.get(name, 0.5)
        if pd.isna(avg_pos):
            avg_pos = 0.5
        # Good openers: classic hits, high energy, typically played early
        score = song.get("combined_score", 0) * 0.3
        score += (1 - avg_pos) * 30  # Favor songs typically played early
        if song.get("is_classic_hit"):
            score += 20
        return score

    def closer_score(song):
        name = song["song_name"]
        avg_pos = pos_lookup.get(name, 0.5)
        if pd.isna(avg_pos):
            avg_pos = 0.5
        # Good closers: high energy anthems, typically played late
        score = song.get("combined_score", 0) * 0.3
        score += avg_pos * 30  # Favor songs typically played late
        if song.get("is_classic_hit"):
            score += 15
        if song.get("is_latest_album"):
            score += 10  # End with something new
        return score

    # Select opener
    others_sorted_opener = sorted(others, key=opener_score, reverse=True)
    opener = others_sorted_opener[0] if others_sorted_opener else None
    if opener:
        others.remove(opener)

    # Select closer
    others_sorted_closer = sorted(others, key=closer_score, reverse=True)
    closer = others_sorted_closer[0] if others_sorted_closer else None
    if closer:
        others.remove(closer)

    # Order middle section - intersperse new and classic
    middle = []
    latest_album_songs = [s for s in others if s.get("is_latest_album")]
    classic_songs = [s for s in others if s.get("is_classic_hit") and not s.get("is_latest_album")]
    other_songs = [s for s in others if s not in latest_album_songs and s not in classic_songs]

    # Sort each group by combined_score
    latest_album_songs.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    classic_songs.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    other_songs.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    # Interleave: classic, new, other, classic, new, other...
    while latest_album_songs or classic_songs or other_songs:
        if classic_songs:
            middle.append(classic_songs.pop(0))
        if latest_album_songs:
            middle.append(latest_album_songs.pop(0))
        if other_songs:
            middle.append(other_songs.pop(0))

    # Add confirmed songs into middle (they're important but flexible placement)
    middle.extend(confirmed)

    # Shuffle middle slightly to avoid predictability but keep energy flow
    # Put higher-energy songs in latter half of middle
    mid_point = len(middle) // 2
    first_half = middle[:mid_point]
    second_half = middle[mid_point:]
    second_half.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    middle = first_half + second_half

    # Assemble final order
    ordered = []
    if opener:
        ordered.append(opener)
    ordered.extend(middle)
    if closer:
        ordered.append(closer)

    # Add position and placement type
    for i, song in enumerate(ordered):
        song["position"] = i + 1
        song["placement_type"] = get_placement_type(i + 1, len(ordered))

    return ordered


def calculate_confidence_score(
    selected: List[Dict],
    constraints_satisfied: Dict[str, bool]
) -> float:
    """Calculate overall confidence in the prediction."""
    score = 0.0

    # Base score from average combined_score of selected songs
    avg_combined = np.mean([s.get("combined_score", 0) for s in selected])
    score += avg_combined * 0.5  # 0-50 points

    # Bonus for constraint satisfaction
    satisfied_count = sum(constraints_satisfied.values())
    total_constraints = len(constraints_satisfied)
    score += (satisfied_count / total_constraints) * 30  # 0-30 points

    # Bonus for confirmed songs
    confirmed_count = sum(1 for s in selected if s.get("is_confirmed", False))
    score += confirmed_count * 5  # 5 points per confirmed song

    # Normalize to 0-100
    return min(100, round(score, 1))


def validate_constraints(selected: List[Dict], total_duration: int) -> Dict[str, bool]:
    """Validate all constraints are satisfied."""

    latest_album_count = sum(1 for s in selected if s.get("is_latest_album"))
    classic_count = sum(1 for s in selected if s.get("is_classic_hit"))
    collab_count = sum(1 for s in selected if s.get("has_featured_artist"))

    return {
        "duration_in_range": SetlistConstraints.MIN_DURATION_SECONDS <= total_duration <= SetlistConstraints.MAX_DURATION_SECONDS,
        "song_count_valid": SetlistConstraints.MIN_SONGS <= len(selected) <= SetlistConstraints.MAX_SONGS,
        "latest_album_minimum": latest_album_count >= SetlistConstraints.MIN_LATEST_ALBUM_SONGS,
        "classic_hits_minimum": classic_count >= SetlistConstraints.MIN_CLASSIC_HITS,
        "collab_limit_respected": collab_count <= SetlistConstraints.MAX_COLLAB_SONGS,
        "has_confirmed_songs": any(s.get("is_confirmed") for s in selected),
    }


def format_duration(seconds: int) -> str:
    """Format seconds as mm:ss."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def generate_predicted_setlist(df: pd.DataFrame) -> PredictedSetlist:
    """Generate the complete predicted setlist."""

    # Get confirmed songs
    confirmed_songs = get_confirmed_songs(df)
    logger.info(f"Confirmed songs: {confirmed_songs}")

    # Select songs using greedy algorithm
    selected, alternates = select_songs_greedy(df, confirmed_songs)

    # Order by energy arc
    ordered = order_setlist_by_energy(selected, df)

    # Calculate total duration
    total_duration = sum(s.get("estimated_duration", 0) for s in ordered)

    # Validate constraints
    constraints_satisfied = validate_constraints(ordered, total_duration)

    # Calculate confidence
    confidence = calculate_confidence_score(ordered, constraints_satisfied)

    # Build result
    result = PredictedSetlist(
        generated_at=datetime.now().isoformat(),
        total_songs=len(ordered),
        total_duration_seconds=total_duration,
        total_duration_formatted=format_duration(total_duration),
        confidence_score=confidence,
        constraints_satisfied=constraints_satisfied,
        setlist=ordered,
        alternates=alternates,
        methodology=(
            "Greedy selection with constraint satisfaction: "
            "1) Include confirmed songs, "
            "2) Ensure latest album representation (2-4 songs), "
            "3) Include classic hits (3+ songs), "
            "4) Respect collab limits (max 3), "
            "5) Optimize for 12-13 min runtime, "
            "6) Order by energy arc (opener → build → peak finale)"
        )
    )

    return result


def print_setlist_summary(result: PredictedSetlist) -> None:
    """Print a summary of the predicted setlist."""

    print("\n" + "=" * 70)
    print("PREDICTED SUPER BOWL LX HALFTIME SETLIST")
    print("Bad Bunny - February 8, 2026")
    print("=" * 70)

    print(f"\nTotal Songs: {result.total_songs}")
    print(f"Total Duration: {result.total_duration_formatted} ({result.total_duration_seconds}s)")
    print(f"Confidence Score: {result.confidence_score}/100")

    print("\n" + "-" * 50)
    print("CONSTRAINT SATISFACTION")
    print("-" * 50)
    for constraint, satisfied in result.constraints_satisfied.items():
        status = "✓" if satisfied else "✗"
        print(f"  {status} {constraint}")

    print("\n" + "-" * 50)
    print("PREDICTED SETLIST")
    print("-" * 50)

    for song in result.setlist:
        pos = song["position"]
        name = song["song_name"]
        dur = format_duration(song["estimated_duration"])
        fmt = song["format"].upper()[:4]
        placement = song["placement_type"].upper()[:6]

        flags = []
        if song.get("is_confirmed"):
            flags.append("CONFIRMED")
        if song.get("is_latest_album"):
            flags.append("NEW")
        if song.get("is_classic_hit"):
            flags.append("HIT")
        if song.get("has_featured_artist"):
            flags.append("FEAT")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"  {pos:>2}. {name:<30} {dur:>5} ({fmt:<4}) {placement:<6}{flag_str}")

    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {result.total_duration_formatted}")

    print("\n" + "-" * 50)
    print("ALTERNATE CANDIDATES")
    print("-" * 50)
    for alt in result.alternates[:5]:
        name = alt["song_name"]
        score = alt["combined_score"]
        print(f"  • {name:<30} (score: {score:.1f})")

    print("\n" + "=" * 70)


def main():
    """Main function to generate predicted setlist."""
    logger.info("=" * 60)
    logger.info("Generating Constrained Super Bowl Setlist Prediction")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading training data...")
    df = load_training_data()
    logger.info(f"Loaded {len(df)} songs")

    # Generate prediction
    logger.info("Generating predicted setlist...")
    result = generate_predicted_setlist(df)

    # Save to JSON (convert numpy types to native Python types)
    output_path = DATA_PROCESSED_DIR / "predicted_setlist.json"

    def convert_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    result_dict = convert_types(asdict(result))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved prediction to: {output_path}")

    # Print summary
    print_setlist_summary(result)


if __name__ == "__main__":
    main()

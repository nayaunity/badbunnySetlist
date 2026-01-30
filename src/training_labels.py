"""Generate training labels for Super Bowl setlist prediction model."""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_song_name(name: str) -> str:
    """Normalize song name for matching."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


# === HIGH-STAKES SHOW IDENTIFIERS ===

# Festival keywords to identify major festival performances
FESTIVAL_KEYWORDS = [
    "coachella", "lollapalooza", "rolling loud", "primavera",
    "governors ball", "bonnaroo", "austin city limits", "acl",
    "electric daisy", "edc", "ultra", "tomorrowland",
    "made in america", "firefly", "outside lands", "life is beautiful",
    "iii points", "hard summer", "day n vegas"
]

# Award show keywords
AWARD_SHOW_KEYWORDS = [
    "grammy", "latin grammy", "billboard", "mtv", "vma", "ama",
    "american music", "bet awards", "premio lo nuestro", "latin ama"
]

# Known flagship shows (venue + date patterns)
FLAGSHIP_SHOWS = {
    # Super Bowl 2020 appearance
    ("hard rock stadium", "02-02-2020"),
    ("hard rock stadium", "2020-02-02"),
    # Add other known important shows by venue
}

# PR Residency finale identifier (last show of residency)
PR_RESIDENCY_FINALE_DATE = "25-01-2026"  # Approximate - will find actual last date


def load_setlist_data() -> Dict[str, Any]:
    """Load Bad Bunny setlist data."""
    path = DATA_RAW_DIR / "bad_bunny_setlists.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_superbowl_data() -> Dict[str, Any]:
    """Load Super Bowl halftime setlist data."""
    path = DATA_RAW_DIR / "superbowl_halftime_setlists.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix v2."""
    path = DATA_PROCESSED_DIR / "feature_matrix_v2.csv"
    return pd.read_csv(path)


def is_festival_show(setlist: Dict[str, Any]) -> bool:
    """Check if a show is a major festival performance."""
    venue_name = setlist.get("venue", {}).get("name", "").lower()
    tour_name = (setlist.get("tour") or "").lower()

    for keyword in FESTIVAL_KEYWORDS:
        if keyword in venue_name or keyword in tour_name:
            return True
    return False


def is_award_show(setlist: Dict[str, Any]) -> bool:
    """Check if a show is an award show performance."""
    venue_name = setlist.get("venue", {}).get("name", "").lower()
    tour_name = (setlist.get("tour") or "").lower()

    for keyword in AWARD_SHOW_KEYWORDS:
        if keyword in venue_name or keyword in tour_name:
            return True
    return False


def is_superbowl_2020(setlist: Dict[str, Any]) -> bool:
    """Check if this is the Super Bowl 2020 appearance."""
    venue_name = setlist.get("venue", {}).get("name", "").lower()
    event_date = setlist.get("event_date", "")

    return "hard rock" in venue_name and "02-02-2020" in event_date


def is_pr_residency_finale(setlist: Dict[str, Any], all_pr_shows: List[Dict]) -> bool:
    """Check if this is the PR residency finale show."""
    if not setlist.get("is_puerto_rico_residency", False):
        return False

    # Find the last PR residency show by date
    event_date = setlist.get("event_date", "")

    # Get all PR show dates and find the latest
    pr_dates = []
    for show in all_pr_shows:
        if show.get("is_puerto_rico_residency", False):
            try:
                d = datetime.strptime(show.get("event_date", ""), "%d-%m-%Y")
                pr_dates.append((d, show.get("event_date")))
            except ValueError:
                pass

    if pr_dates:
        latest_date = max(pr_dates, key=lambda x: x[0])[1]
        return event_date == latest_date

    return False


def identify_flagship_shows(setlists: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Identify flagship/high-stakes shows from the setlist data.

    Returns:
        - flagship_shows: list of high-stakes show setlists
        - pr_residency_shows: list of PR residency shows (for holdout)
    """
    flagship_shows = []
    pr_residency_shows = []

    # First pass: collect PR residency shows
    for setlist in setlists:
        if setlist.get("is_puerto_rico_residency", False):
            pr_residency_shows.append(setlist)

    logger.info(f"Found {len(pr_residency_shows)} PR residency shows")

    # Second pass: identify flagship shows
    for setlist in setlists:
        is_flagship = False
        flagship_type = None

        if is_festival_show(setlist):
            is_flagship = True
            flagship_type = "festival"
        elif is_award_show(setlist):
            is_flagship = True
            flagship_type = "award_show"
        elif is_superbowl_2020(setlist):
            is_flagship = True
            flagship_type = "superbowl_2020"
        elif is_pr_residency_finale(setlist, pr_residency_shows):
            is_flagship = True
            flagship_type = "pr_finale"

        if is_flagship:
            setlist["_flagship_type"] = flagship_type
            flagship_shows.append(setlist)

    # Log findings
    flagship_types = defaultdict(int)
    for show in flagship_shows:
        flagship_types[show.get("_flagship_type", "unknown")] += 1

    logger.info(f"Identified {len(flagship_shows)} flagship shows:")
    for ftype, count in flagship_types.items():
        logger.info(f"  - {ftype}: {count}")

    return flagship_shows, pr_residency_shows


def calculate_flagship_appearance(
    normalized_name: str,
    flagship_shows: List[Dict[str, Any]]
) -> Tuple[int, int, float]:
    """
    Calculate flagship show appearance metrics for a song.

    Returns:
        - appeared_count: number of flagship shows song appeared in
        - total_flagship: total number of flagship shows
        - appearance_rate: appeared_count / total_flagship
    """
    appeared_count = 0
    total_flagship = len(flagship_shows)

    for show in flagship_shows:
        songs = show.get("songs", [])
        song_names = [normalize_song_name(s.get("name", "")) for s in songs]
        if normalized_name in song_names:
            appeared_count += 1

    appearance_rate = appeared_count / total_flagship if total_flagship > 0 else 0
    return appeared_count, total_flagship, appearance_rate


def analyze_superbowl_patterns(superbowl_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze patterns from historical Super Bowl halftime shows.

    Returns metrics about typical halftime setlists.
    """
    shows = superbowl_data.get("shows", [])
    shows_with_data = [s for s in shows if s.get("setlist_found", False)]

    if not shows_with_data:
        logger.warning("No Super Bowl setlist data available for pattern analysis")
        return {}

    # Analyze patterns
    song_counts = [s.get("song_count", 0) for s in shows_with_data]
    guest_counts = [len(s.get("guest_artists", [])) for s in shows_with_data]

    patterns = {
        "avg_songs": np.mean(song_counts),
        "min_songs": min(song_counts),
        "max_songs": max(song_counts),
        "avg_guests": np.mean(guest_counts),
        "shows_analyzed": len(shows_with_data),
    }

    logger.info("Super Bowl Halftime Patterns:")
    logger.info(f"  - Average songs: {patterns['avg_songs']:.1f}")
    logger.info(f"  - Range: {patterns['min_songs']}-{patterns['max_songs']} songs")
    logger.info(f"  - Average guest appearances: {patterns['avg_guests']:.1f}")

    return patterns


def calculate_superbowl_likelihood_score(
    row: pd.Series,
    sb_patterns: Dict[str, float],
    max_performances: int
) -> float:
    """
    Calculate a Super Bowl likelihood score based on how well a song
    fits historical halftime patterns.

    Factors:
    1. Performance popularity (songs that get played a lot = hits)
    2. Recency (recent songs often featured)
    3. Duration (shorter songs fit medleys better)
    4. Solo vs collab (fewer collabs = easier logistics)
    5. Cultural significance (meaningful moments)
    """
    score = 0.0

    # 1. Performance popularity (0-30 points)
    # Top performed songs are likely biggest hits
    perf_freq = row.get("performance_frequency", 0)
    popularity_score = min(30, perf_freq * 100)  # Scale to 0-30
    score += popularity_score

    # 2. Recency bonus (0-20 points)
    # Latest album songs get a boost (artists promote new material)
    if row.get("is_latest_album", 0) == 1:
        score += 20
    elif row.get("release_year", 0) >= 2024:
        score += 15
    elif row.get("release_year", 0) >= 2022:
        score += 10
    elif row.get("release_year", 0) >= 2020:
        score += 5

    # 3. Duration fit (0-15 points)
    # Shorter songs work better in medleys
    duration = row.get("duration_seconds", 240)
    if duration > 0:
        if duration <= 180:
            score += 15  # Short - perfect for medley
        elif duration <= 210:
            score += 12
        elif duration <= 240:
            score += 8
        elif duration <= 270:
            score += 4
        # Longer songs get no bonus

    # 4. Solo performance advantage (0-10 points)
    # Songs without featured artists are logistically easier
    if row.get("has_featured_artist", 0) == 0:
        score += 10
    elif row.get("major_collab_artist", 0) == 1:
        score += 5  # Major collabs might bring surprise guests

    # 5. Cultural significance (0-15 points)
    # High cultural significance for hometown pride
    cultural = row.get("cultural_significance", 0)
    score += cultural * 5  # 0, 5, 10, or 15 points

    # 6. Medley-friendly flag (0-10 points)
    if row.get("medley_friendly", 0) == 1:
        score += 10

    # 7. PR Residency signal (0-10 points)
    # Songs played at PR residency = recently rehearsed
    if row.get("performed_at_pr_residency", 0) == 1:
        pr_freq = row.get("pr_residency_frequency", 0)
        score += min(10, pr_freq * 15)

    # 8. Promotional signal bonus (0-10 points)
    if row.get("in_halftime_trailer", 0) == 1:
        score += 10

    # 9. Penalty for already performed at Super Bowl
    if row.get("performed_superbowl_2020", 0) == 1:
        score -= 15  # Likely won't repeat

    # Normalize to 0-100 scale
    # Max possible: 30+20+15+10+15+10+10+10 = 120 (minus penalty)
    score = max(0, min(100, score))

    return round(score, 2)


def calculate_pr_holdout_labels(
    normalized_name: str,
    pr_shows: List[Dict[str, Any]]
) -> Tuple[int, float]:
    """
    Calculate labels based on PR residency appearances (holdout set).

    Returns:
        - appeared_in_pr: binary flag
        - pr_appearance_rate: % of PR shows song appeared in
    """
    appeared_count = 0
    total_shows = len(pr_shows)

    for show in pr_shows:
        songs = show.get("songs", [])
        song_names = [normalize_song_name(s.get("name", "")) for s in songs]
        if normalized_name in song_names:
            appeared_count += 1

    appeared = 1 if appeared_count > 0 else 0
    rate = appeared_count / total_shows if total_shows > 0 else 0

    return appeared, rate


def create_training_labels(
    df: pd.DataFrame,
    flagship_shows: List[Dict],
    pr_shows: List[Dict],
    sb_patterns: Dict[str, float]
) -> pd.DataFrame:
    """Create training labels combining both approaches."""

    # Create normalized name for matching
    df["_normalized"] = df["song_name"].apply(normalize_song_name)

    # Get max performances for normalization
    max_performances = df["times_performed_live"].max()

    # === APPROACH 1: Flagship Show Appearance ===
    logger.info("Calculating flagship show appearances...")

    flagship_metrics = df["_normalized"].apply(
        lambda x: calculate_flagship_appearance(x, flagship_shows)
    )

    df["flagship_appearances"] = flagship_metrics.apply(lambda x: x[0])
    df["flagship_shows_total"] = flagship_metrics.apply(lambda x: x[1])
    df["flagship_appearance_rate"] = flagship_metrics.apply(lambda x: x[2])
    df["appeared_in_flagship_show"] = (df["flagship_appearances"] > 0).astype(int)

    # === APPROACH 2: Super Bowl Likelihood Score ===
    logger.info("Calculating Super Bowl likelihood scores...")

    df["superbowl_likelihood_score"] = df.apply(
        lambda row: calculate_superbowl_likelihood_score(row, sb_patterns, max_performances),
        axis=1
    )

    # === COMBINED TARGET VARIABLE ===
    # Weight both signals to create final prediction target
    # Flagship appearance (40%) + SB likelihood (60%)

    # Normalize flagship appearance rate to 0-100
    flagship_normalized = df["flagship_appearance_rate"] * 100

    df["combined_score"] = (
        0.4 * flagship_normalized +
        0.6 * df["superbowl_likelihood_score"]
    ).round(2)

    # Create binary target: top candidates
    # Flag songs in top 15% of combined score as likely setlist candidates
    threshold = df["combined_score"].quantile(0.85)
    df["likely_setlist_candidate"] = (df["combined_score"] >= threshold).astype(int)

    # === HOLDOUT VALIDATION LABELS (PR Residency) ===
    logger.info("Calculating PR residency holdout labels...")

    pr_metrics = df["_normalized"].apply(
        lambda x: calculate_pr_holdout_labels(x, pr_shows)
    )

    df["holdout_appeared_in_pr"] = pr_metrics.apply(lambda x: x[0])
    df["holdout_pr_appearance_rate"] = pr_metrics.apply(lambda x: round(x[1], 4))

    # Drop helper column
    df = df.drop("_normalized", axis=1)

    return df


def print_label_summary(df: pd.DataFrame, flagship_shows: List[Dict], pr_shows: List[Dict]) -> None:
    """Print summary statistics for the training labels."""

    print("\n" + "=" * 70)
    print("TRAINING LABELS SUMMARY")
    print("=" * 70)

    # === Approach 1: Flagship Shows ===
    print("\n" + "-" * 50)
    print("APPROACH 1: FLAGSHIP SHOW APPEARANCES")
    print("-" * 50)

    print(f"\nTotal flagship shows identified: {len(flagship_shows)}")

    flagship_songs = df[df["appeared_in_flagship_show"] == 1]
    print(f"Songs that appeared in flagship shows: {len(flagship_songs)} / {len(df)}")

    print("\nTop 15 songs by flagship appearance rate:")
    top_flagship = df.nlargest(15, "flagship_appearance_rate")[
        ["song_name", "flagship_appearances", "flagship_appearance_rate", "times_performed_live"]
    ]
    for _, row in top_flagship.iterrows():
        print(f"  {row['song_name']:<30} {row['flagship_appearances']:>2}/{len(flagship_shows)} ({row['flagship_appearance_rate']*100:>5.1f}%)")

    # === Approach 2: SB Likelihood ===
    print("\n" + "-" * 50)
    print("APPROACH 2: SUPER BOWL LIKELIHOOD SCORE")
    print("-" * 50)

    print(f"\nScore distribution:")
    print(f"  Mean: {df['superbowl_likelihood_score'].mean():.1f}")
    print(f"  Std:  {df['superbowl_likelihood_score'].std():.1f}")
    print(f"  Min:  {df['superbowl_likelihood_score'].min():.1f}")
    print(f"  Max:  {df['superbowl_likelihood_score'].max():.1f}")

    print("\nTop 15 songs by Super Bowl likelihood:")
    top_sb = df.nlargest(15, "superbowl_likelihood_score")[
        ["song_name", "superbowl_likelihood_score", "is_latest_album", "performance_frequency"]
    ]
    for _, row in top_sb.iterrows():
        new_flag = " [NEW]" if row["is_latest_album"] == 1 else ""
        print(f"  {row['song_name']:<30} {row['superbowl_likelihood_score']:>5.1f}{new_flag}")

    # === Combined Score ===
    print("\n" + "-" * 50)
    print("COMBINED TARGET VARIABLE")
    print("-" * 50)

    likely_count = df["likely_setlist_candidate"].sum()
    print(f"\nLikely setlist candidates (top 15%): {likely_count} songs")

    print("\nTop 20 predicted setlist candidates:")
    top_combined = df.nlargest(20, "combined_score")[
        ["song_name", "combined_score", "superbowl_likelihood_score",
         "flagship_appearance_rate", "is_latest_album"]
    ]
    for i, (_, row) in enumerate(top_combined.iterrows(), 1):
        new_flag = "*" if row["is_latest_album"] == 1 else " "
        print(f"  {i:>2}. {row['song_name']:<30} {row['combined_score']:>5.1f} "
              f"(SB:{row['superbowl_likelihood_score']:>4.0f}, Flag:{row['flagship_appearance_rate']*100:>4.0f}%) {new_flag}")

    # === Holdout Validation ===
    print("\n" + "-" * 50)
    print("HOLDOUT VALIDATION (PR RESIDENCY)")
    print("-" * 50)

    print(f"\nPR Residency shows (holdout set): {len(pr_shows)}")
    pr_songs = df[df["holdout_appeared_in_pr"] == 1]
    print(f"Songs performed at PR residency: {len(pr_songs)} / {len(df)}")

    # Validate: Do our top predictions match PR residency?
    top_20_predictions = set(df.nlargest(20, "combined_score")["song_name"])
    pr_performed = set(df[df["holdout_appeared_in_pr"] == 1]["song_name"])
    overlap = top_20_predictions & pr_performed

    print(f"\nValidation: Top 20 predictions vs PR residency:")
    print(f"  Overlap: {len(overlap)} / 20 ({len(overlap)/20*100:.0f}%)")
    print(f"  Songs in both: {', '.join(list(overlap)[:5])}...")

    # === Label Distribution ===
    print("\n" + "-" * 50)
    print("LABEL DISTRIBUTION")
    print("-" * 50)

    print(f"\nappeared_in_flagship_show: {df['appeared_in_flagship_show'].sum()} / {len(df)} ({df['appeared_in_flagship_show'].mean()*100:.1f}%)")
    print(f"likely_setlist_candidate:  {df['likely_setlist_candidate'].sum()} / {len(df)} ({df['likely_setlist_candidate'].mean()*100:.1f}%)")
    print(f"holdout_appeared_in_pr:    {df['holdout_appeared_in_pr'].sum()} / {len(df)} ({df['holdout_appeared_in_pr'].mean()*100:.1f}%)")

    print("\n" + "=" * 70)


def main():
    """Main function to generate training labels."""
    logger.info("=" * 60)
    logger.info("Generating Training Labels for Setlist Prediction")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    setlist_data = load_setlist_data()
    superbowl_data = load_superbowl_data()
    df = load_feature_matrix()

    logger.info(f"Loaded {len(df)} songs from feature matrix")

    setlists = setlist_data.get("setlists", [])
    logger.info(f"Loaded {len(setlists)} setlists")

    # Identify flagship shows
    logger.info("Identifying flagship shows...")
    flagship_shows, pr_shows = identify_flagship_shows(setlists)

    # Analyze Super Bowl patterns
    logger.info("Analyzing Super Bowl halftime patterns...")
    sb_patterns = analyze_superbowl_patterns(superbowl_data)

    # Create training labels
    logger.info("Creating training labels...")
    df = create_training_labels(df, flagship_shows, pr_shows, sb_patterns)

    # Save training data
    output_path = DATA_PROCESSED_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to: {output_path}")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Print summary
    print_label_summary(df, flagship_shows, pr_shows)


if __name__ == "__main__":
    main()

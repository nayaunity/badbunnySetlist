"""Add contextual Super Bowl-specific features to the feature matrix."""
import logging
import re
from pathlib import Path
from typing import Set

import pandas as pd

from .config import DATA_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_for_matching(name: str) -> str:
    """Normalize song name for matching."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


# === MANUAL DATA ===

# Songs confirmed in halftime trailer (Apple Music promo)
TRAILER_SONGS = {
    "baile inolvidable",
}

# Cultural significance ratings (0-3 scale)
# 3 = High cultural/political significance (Puerto Rico pride, political themes)
# 2 = Medium significance (important cultural moments)
# 1 = Low significance (mild cultural references)
# 0 = No particular cultural significance
CULTURAL_SIGNIFICANCE = {
    # High significance (3) - Puerto Rico pride/political themes
    "el apagon": 3,
    "nuevayol": 3,
    "afilando los cuchillos": 3,
    "booker t": 3,  # References Puerto Rico issues
    "el apagon live": 3,

    # Medium significance (2)
    "la santa": 2,
    "yo perreo sola": 2,  # Feminist anthem
    "andrea": 2,  # LGBTQ+ themes
    "caro": 2,
    "bichayal": 2,

    # Lower significance (1)
    "dakiti": 1,
    "callaita": 1,
}

# Song performed at Super Bowl LIV (2020) with Shakira/JLo
# Bad Bunny performed "I Like It" and "Chantaje" medley
SUPERBOWL_2020_SONGS = {
    "i like it",
    "chantaje",
}

# Songs with significant English lyrics
ENGLISH_LYRICS_SONGS = {
    "mia",  # With Drake
    "i like it",  # With Cardi B
    "im the one",  # DJ Khaled track
    "krippy kush",  # Has English hook
    "te bote",  # Some English
    "callaita",  # Some English
    "la noche de anoche",  # Some English
}

# Major global collaborations (globally recognized artists)
MAJOR_COLLAB_ARTISTS = {
    "drake",
    "j balvin",
    "daddy yankee",
    "cardi b",
    "dj khaled",
    "travis scott",
    "post malone",
    "rosalia",
    "jhay cortez",
    "anuel aa",
    "ozuna",
    "becky g",
    "nicky jam",
    "sech",
    "myke towers",
    "farruko",
    "arcangel",
}

# Songs known to have major collabs (for when artist list doesn't match exactly)
MAJOR_COLLAB_SONGS = {
    "mia": "drake",
    "i like it": "cardi b",
    "im the one": "dj khaled",
    "que pretendes": "j balvin",
    "oasis": "j balvin",
    "un dia one day": "j balvin, dua lipa, tainy",
    "la santa": "daddy yankee",
    "dakiti": "jhay cortez",
    "yo perreo sola": None,  # Solo but iconic
    "callaita": "tainy",
    "te bote": "ozuna, nicky jam",
    "safaera": "jowell y randy, nengo flow",
    "la romana": "el alfa",
    "efecto": None,
    "moscow mule": None,
    "titi me pregunto": None,
    "me porto bonito": "chencho corleone",
    "ojitos lindos": "bomba estereo",
}


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contextual Super Bowl features to the dataframe."""

    # Create normalized name column for matching
    df["_normalized"] = df["song_name"].apply(normalize_for_matching)

    # === 1. Promotional Signals ===

    # In halftime trailer
    df["in_halftime_trailer"] = df["_normalized"].apply(
        lambda x: 1 if x in TRAILER_SONGS else 0
    )

    # Placeholder for confirmed songs (news before Feb 8)
    df["confirmed_for_halftime"] = 0

    # === 2. Cultural Significance Score ===

    df["cultural_significance"] = df["_normalized"].apply(
        lambda x: CULTURAL_SIGNIFICANCE.get(x, 0)
    )

    # Flag Super Bowl 2020 performance
    df["performed_superbowl_2020"] = df["_normalized"].apply(
        lambda x: 1 if x in SUPERBOWL_2020_SONGS else 0
    )

    # === 3. Crossover/Mainstream Appeal ===

    # Has English lyrics
    df["has_english_lyrics"] = df["_normalized"].apply(
        lambda x: 1 if x in ENGLISH_LYRICS_SONGS else 0
    )

    # Major collab artist detection
    def has_major_collab(row):
        normalized = row["_normalized"]

        # Check if in known collab songs
        if normalized in MAJOR_COLLAB_SONGS:
            collab = MAJOR_COLLAB_SONGS[normalized]
            if collab:
                return 1

        # Check all_artists column if it exists (it doesn't in the CSV, but check anyway)
        # We'll check the num_featured_artists and song names
        if row.get("has_featured_artist", 0) == 1:
            # Check if any major artist in the collaboration
            # This is a heuristic - in production we'd have the actual artist list
            if normalized in MAJOR_COLLAB_SONGS:
                return 1

        return 0

    df["major_collab_artist"] = df.apply(has_major_collab, axis=1)

    # Also add based on known major collabs
    for song, artist in MAJOR_COLLAB_SONGS.items():
        if artist:
            mask = df["_normalized"] == song
            df.loc[mask, "major_collab_artist"] = 1

    # === 4. Super Bowl Format Fit ===

    # Medley-friendly placeholder (for manual review)
    # Pre-populate with songs that have high performance frequency
    # and are in "middle" position (versatile songs)
    df["medley_friendly"] = None  # Will be manually reviewed

    # Pre-flag likely candidates based on:
    # - High performance frequency (> 0.10)
    # - Has been performed live
    # - Not too long (< 300 seconds)
    medley_candidates = (
        (df["performance_frequency"] > 0.10) &
        (df["ever_performed_live"] == 1) &
        (df["duration_seconds"] < 300)
    )
    df.loc[medley_candidates, "medley_friendly"] = 1

    # Drop helper column
    df = df.drop("_normalized", axis=1)

    return df


def print_flagged_songs(df: pd.DataFrame) -> None:
    """Print summary of songs flagged for each contextual feature."""

    print("\n" + "=" * 70)
    print("CONTEXTUAL FEATURES SUMMARY")
    print("=" * 70)

    # === Promotional Signals ===
    print("\n" + "-" * 50)
    print("1. PROMOTIONAL SIGNALS")
    print("-" * 50)

    trailer_songs = df[df["in_halftime_trailer"] == 1]["song_name"].tolist()
    print(f"\nIn Halftime Trailer ({len(trailer_songs)}):")
    for song in trailer_songs:
        print(f"  • {song}")

    print("\nConfirmed for Halftime: (placeholder column - populate manually)")

    # === Cultural Significance ===
    print("\n" + "-" * 50)
    print("2. CULTURAL SIGNIFICANCE")
    print("-" * 50)

    for score in [3, 2, 1]:
        songs = df[df["cultural_significance"] == score][["song_name", "times_performed_live"]].values.tolist()
        labels = {3: "High (Puerto Rico pride/political)", 2: "Medium", 1: "Lower"}
        print(f"\nScore {score} - {labels[score]} ({len(songs)}):")
        for song, times in songs:
            print(f"  • {song} ({times} performances)")

    sb2020 = df[df["performed_superbowl_2020"] == 1]["song_name"].tolist()
    print(f"\nPerformed at Super Bowl 2020 ({len(sb2020)}):")
    for song in sb2020:
        print(f"  • {song} (may want to exclude - already performed at SB)")

    # === Crossover Appeal ===
    print("\n" + "-" * 50)
    print("3. CROSSOVER/MAINSTREAM APPEAL")
    print("-" * 50)

    english = df[df["has_english_lyrics"] == 1][["song_name", "times_performed_live"]].values.tolist()
    print(f"\nHas English Lyrics ({len(english)}):")
    for song, times in english:
        print(f"  • {song} ({times} performances)")

    collabs = df[df["major_collab_artist"] == 1][["song_name", "times_performed_live"]].values.tolist()
    print(f"\nMajor Collab Artist ({len(collabs)}):")
    for song, times in sorted(collabs, key=lambda x: -x[1])[:15]:
        print(f"  • {song} ({times} performances)")
    if len(collabs) > 15:
        print(f"  ... and {len(collabs) - 15} more")

    # === Super Bowl Format Fit ===
    print("\n" + "-" * 50)
    print("4. SUPER BOWL FORMAT FIT")
    print("-" * 50)

    medley = df[df["medley_friendly"] == 1].sort_values("times_performed_live", ascending=False)
    print(f"\nMedley-Friendly Candidates ({len(medley)}) - TOP 30:")
    print("(Auto-flagged: high frequency, performed live, < 5 min duration)")
    print("-" * 50)
    for _, row in medley.head(30).iterrows():
        flags = []
        if row["is_latest_album"] == 1:
            flags.append("NEW")
        if row["cultural_significance"] >= 2:
            flags.append("CULTURAL")
        if row["major_collab_artist"] == 1:
            flags.append("COLLAB")
        if row["performed_at_pr_residency"] == 1:
            flags.append("PR")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  • {row['song_name']:<30} ({row['times_performed_live']:>3} times, {row['duration_seconds']:>3}s){flag_str}")

    # === Feature Distribution ===
    print("\n" + "-" * 50)
    print("FEATURE DISTRIBUTION")
    print("-" * 50)

    contextual_cols = [
        "in_halftime_trailer", "confirmed_for_halftime", "cultural_significance",
        "performed_superbowl_2020", "has_english_lyrics", "major_collab_artist"
    ]

    for col in contextual_cols:
        if col == "cultural_significance":
            counts = df[col].value_counts().sort_index()
            print(f"\n{col}:")
            for val, count in counts.items():
                print(f"  {val}: {count}")
        else:
            count = df[col].sum()
            print(f"{col}: {count} / {len(df)} ({count/len(df)*100:.1f}%)")

    medley_count = df["medley_friendly"].sum()
    print(f"medley_friendly (auto-flagged): {medley_count} / {len(df)}")

    print("\n" + "=" * 70)


def main():
    """Main function to add contextual features."""
    logger.info("=" * 60)
    logger.info("Adding Contextual Super Bowl Features")
    logger.info("=" * 60)

    # Load existing feature matrix
    input_path = DATA_PROCESSED_DIR / "feature_matrix.csv"
    logger.info(f"Loading feature matrix from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} songs")

    # Add contextual features
    logger.info("Adding contextual features...")
    df = add_contextual_features(df)

    # Save updated matrix
    output_path = DATA_PROCESSED_DIR / "feature_matrix_v2.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved updated matrix to: {output_path}")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Print summary
    print_flagged_songs(df)


if __name__ == "__main__":
    main()

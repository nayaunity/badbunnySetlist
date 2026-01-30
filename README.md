# Bad Bunny Super Bowl LX Halftime Setlist Predictor

A machine learning model to predict Bad Bunny's setlist for the Super Bowl LX halftime show on February 8, 2026.

## Problem Statement

This project frames setlist prediction as a **binary classification problem**: for each song in Bad Bunny's catalog, we predict P(song appears in setlist) — the probability that a given song will be performed during the halftime show.

## Constraints

### Performance Time
- Halftime shows are approximately **12-13 minutes** of actual performance time

### Historical Setlist Patterns
- Recent halftime shows feature medleys of **8 to 12 songs**:
  - Kendrick Lamar (2025): 10 songs
  - Usher (2024): 12 songs
- Songs are often shortened or performed as snippets within medleys
- Guest performers are common and can add additional song appearances

### Cultural Context
- This is Bad Bunny's **only scheduled US performance** in 2026, adding high cultural significance
- The Super Bowl audience is diverse, including casual listeners and dedicated fans
- Song selection likely balances mainstream hits with cultural representation

## Output

The model produces a **ranked list of songs with probability scores**, which is then filtered to a feasible setlist of 8-12 songs based on:
- Probability threshold
- Performance time constraints
- Medley compatibility

## Project Structure

```
badbunnySetlist/
├── README.md
├── data/
│   ├── raw/           # Original, unprocessed data
│   └── processed/     # Cleaned and feature-engineered data
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/               # Source code and scripts
└── models/            # Trained model artifacts
```

## Data Sources (Planned)

- Spotify streaming data and song metadata
- Historical setlist data from previous tours
- Billboard chart performance
- Social media engagement metrics
- Previous Super Bowl halftime setlist patterns

## Getting Started

*Coming soon*

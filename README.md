# Bad Bunny Super Bowl LX Halftime Setlist Predictor

**Status:** Prediction made, awaiting Super Bowl LX (February 8, 2026)

A machine learning project that predicts Bad Bunny's setlist for the Super Bowl LX halftime show using historical performance data, constraint optimization, and external signal validation.

---

## The Prediction

**Expected Setlist (10 songs, 12:59 runtime)**

| # | Song | Confidence | Type |
|---|------|------------|------|
| 1 | BAILE INoLVIDABLE | 100% | Confirmed |
| 2 | Yo perreo sola | 100% | Classic |
| 3 | Amorfoda | 98% | Classic |
| 4 | Chambea | 98% | Classic |
| 5 | Si estuviésemos juntos | 96% | Classic |
| 6 | NUEVAYoL | 95% | New Album |
| 7 | Soy peor | 95% | Classic |
| 8 | La santa | 94% | Classic |
| 9 | Me porto bonito | 94% | Classic |
| 10 | Estamos bien | 94% | Classic |

**Model Confidence:** 96% | **Wild Card:** Cardi B guest appearance (40%)

[**View Full Prediction Report**](data/final/PREDICTION_REPORT.md)

---

## Problem Statement

This project frames setlist prediction as a **binary classification problem**: for each of Bad Bunny's 342 songs, we predict P(song appears in setlist) — the probability of appearing in the halftime show.

### Constraints
- **Runtime:** 12-14 minutes of performance time
- **Song Count:** 8-12 songs (medley format)
- **Album Balance:** Mix of new album + classic hits
- **Guest Logistics:** Limited featured artist appearances

---

## Methodology

### Data Pipeline

```
Setlist.fm API ──▶ Feature Engineering ──▶ ML Model ──▶ Constraint Optimizer
(471 setlists)     (26 features)          (LogReg)     (greedy selection)
                                              │
MusicBrainz ───────────────────────────────────┘
(342 songs)
```

### Key Features
1. **medley_friendly** — Songs that transition well in live sets
2. **position_category** — Typical placement (opener/closer)
3. **is_primary_artist** — Solo vs. featured tracks
4. **cultural_significance** — Puerto Rico pride, political themes
5. **performance_frequency** — How often played live

### Model Performance
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.834 |
| F1 Score | 0.714 |
| Holdout Accuracy | 90% |

---

## Project Structure

```
badbunnySetlist/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # API response data
│   │   ├── bad_bunny_setlists.json
│   │   ├── bad_bunny_catalog.json
│   │   └── superbowl_halftime_setlists.json
│   ├── processed/              # Feature matrices
│   │   ├── feature_matrix.csv
│   │   ├── training_data.csv
│   │   ├── model_predictions.csv
│   │   └── predicted_setlist.json
│   └── final/                  # Deliverables
│       ├── PREDICTION_REPORT.md
│       ├── setlist_prediction_visual.html
│       ├── predicted_setlists.json
│       ├── predicted_setlists_validated.json
│       └── evaluation_template.json
├── src/
│   ├── config.py               # API keys, constants
│   ├── api_client.py           # Setlist.fm wrapper
│   ├── collectors/             # Data collection
│   │   ├── bad_bunny.py
│   │   ├── superbowl.py
│   │   └── musicbrainz.py
│   ├── features.py             # Feature engineering
│   ├── contextual_features.py  # Super Bowl-specific
│   ├── training_labels.py      # Label generation
│   ├── train_model.py          # ML training
│   ├── setlist_optimizer.py    # Constraint optimization
│   ├── final_prediction.py     # Ensemble predictions
│   └── validate_prediction.py  # External validation
├── models/
│   ├── setlist_predictor.pkl   # Trained model
│   └── model_card.json         # Performance metadata
└── notebooks/                  # Exploration (optional)
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- Setlist.fm API key

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/badbunnySetlist.git
cd badbunnySetlist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your SETLISTFM_API_KEY
```

### Reproduce the Analysis

```bash
# 1. Collect data (requires API key)
python -m src.collect_data

# 2. Build features
python -m src.features
python -m src.contextual_features
python -m src.training_labels

# 3. Train model
python -m src.train_model

# 4. Generate predictions
python -m src.setlist_optimizer
python -m src.final_prediction
python -m src.validate_prediction
```

---

## Final Deliverables

| File | Description |
|------|-------------|
| [PREDICTION_REPORT.md](data/final/PREDICTION_REPORT.md) | Complete prediction analysis |
| [setlist_prediction_visual.html](data/final/setlist_prediction_visual.html) | Social media visual |
| [predicted_setlists_validated.json](data/final/predicted_setlists_validated.json) | Machine-readable predictions |
| [evaluation_template.json](data/final/evaluation_template.json) | Post-event scoring template |

---

## Evaluation (After Feb 8, 2026)

### Success Criteria
| Rating | Criteria |
|--------|----------|
| Excellent | 8+ songs correct, guest prediction right |
| Good | 6-7 songs correct |
| Moderate | 4-5 songs correct |
| Poor | 2-3 songs correct |

### Metrics
```
Precision = Correct Predictions / Total Predictions
Recall = Correct Predictions / Actual Setlist Size
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## Data Sources

- **[Setlist.fm](https://www.setlist.fm)** — 471 Bad Bunny setlists (2016-2025)
- **[MusicBrainz](https://musicbrainz.org)** — 342 songs with metadata
- **Super Bowl halftimes** — 9 shows (2015-2024) for format analysis

---

## Technologies

- **Python 3.11** — Core language
- **pandas** — Data manipulation
- **scikit-learn** — ML modeling
- **Setlist.fm API** — Performance data
- **MusicBrainz API** — Catalog data

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Setlist.fm for comprehensive setlist data
- MusicBrainz for open music metadata
- Bad Bunny fans who documented performance history

---

*Prediction generated January 30, 2026. Results will be validated after Super Bowl LX.*

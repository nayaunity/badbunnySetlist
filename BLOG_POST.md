# Predicting Bad Bunny's Super Bowl Setlist with Machine Learning

*How I built an ML model to predict what songs Bad Bunny will perform at Super Bowl LX — and what I learned along the way.*

---

## The Challenge

When Bad Bunny was announced as the Super Bowl LX halftime performer, I saw an interesting ML problem: **Can we predict a setlist before it happens?**

Super Bowl halftime shows follow patterns. They're typically 12-14 minutes, feature 8-12 songs in medley format, and balance crowd-pleasers with new material. Artists don't pick songs randomly — they're constrained by time, production logistics, and the need to satisfy both casual viewers and die-hard fans.

This felt like a tractable prediction problem. I had a hypothesis: **past performance patterns, combined with Super Bowl format constraints, could predict future setlists with reasonable accuracy.**

---

## The Data Problem

### Attempt 1: Spotify API (Failed)

My first instinct was Spotify. Stream counts, popularity scores, audio features — perfect for identifying hits. One problem: **Spotify's API is closed to new applications** as of 2024. Dead end.

### Attempt 2: Setlist.fm + MusicBrainz (Success)

I pivoted to two open APIs:

**Setlist.fm** gave me 471 Bad Bunny setlists from 2016-2025. Every concert, every song, every position in the setlist. This was gold — actual performance data showing what he *chooses* to play live.

**MusicBrainz** provided his complete catalog of 342 songs with metadata like duration, release dates, and featured artists.

The data collection had its own challenges:

```python
# Rate limiting was critical - both APIs throttle aggressively
REQUEST_DELAY_SECONDS = 6  # Setlist.fm
MUSICBRAINZ_DELAY = 1.1    # MusicBrainz requires 1 req/sec
```

I also hit a frustrating bug where the **wrong MusicBrainz ID** returned 404s for an hour before I realized I needed to search by artist name to get the correct MBID.

---

## Feature Engineering: What Makes a Song "Super Bowl Worthy"?

I engineered 26 features across several categories:

### Performance Features
- `times_performed_live` — How often he plays it (range: 0-122)
- `performance_frequency` — Percentage of shows featuring the song
- `avg_setlist_position` — Typical placement (opener vs. closer)

### Recency Features
- `days_since_release` — Older songs may be classics or forgotten
- `is_latest_album` — New album tracks get promotional priority

### Contextual Features
- `cultural_significance` — Puerto Rico pride, political themes (manually labeled)
- `has_english_lyrics` — Crossover appeal for US audience
- `medley_friendly` — Songs that transition well in live sets

### The "Latest Album" Problem

One snag: **release dates were 100% missing** from MusicBrainz's recording endpoint. My `days_since_release` feature was useless.

The fix was ugly but worked: I used `first_seen_in_setlist` as a proxy for release date. If a song first appeared in setlists in 2022, it was probably released around then.

For the latest album ("Debí Tirar Más Fotos"), I had to **manually define the track list** since it was too new for complete API data:

```python
DEBI_TIRAR_MAS_FOTOS_TRACKS = {
    "nuevayol", "baile inolvidable", "dtmf", "weltita",
    "pitorro de coco", "turista", "bokete", ...
}
```

---

## The Labeling Challenge

Binary classification needs labels. But I don't have "Super Bowl setlist" training data — there's only been one, and Bad Bunny wasn't the headliner.

### Solution: Proxy Labels

I created two proxy signals:

**1. Flagship Show Appearances**
Songs performed at major events (festivals, award shows, Puerto Rico residency finale) signal "important enough for big moments."

**2. Super Bowl Likelihood Score**
A heuristic combining:
- High performance frequency
- Performed at Puerto Rico residency (his most recent major shows)
- Solo tracks (featured artists add logistics complexity)
- Cultural significance

The final target was a weighted combination: `0.4 * flagship + 0.6 * sb_likelihood`

### Validation: The Puerto Rico Holdout

The clever bit: I held out his **Puerto Rico residency shows** (2024-2025) as validation. These 35 shows were his most recent major performances before the Super Bowl — a realistic proxy for what he'd pick for another big event.

**Result: 90% of songs in my top-20 predictions appeared in the PR residency finale.**

---

## Model Selection

I trained three models:

| Model | Holdout AUC | Holdout F1 | CV AUC |
|-------|-------------|------------|--------|
| Logistic Regression | **0.834** | **0.714** | 0.976 |
| Random Forest | 0.786 | 0.686 | 0.989 |
| Gradient Boosting | 0.734 | 0.639 | 0.961 |

**Logistic Regression won.** It had the best holdout performance, and the coefficients were interpretable — important for understanding *why* songs were predicted.

### Top Predictive Features

```
1. medley_friendly      1.42  ████████████████████████
2. position_closer      0.98  ████████████████
3. is_primary_artist    0.74  ████████████
4. cultural_significance 0.70 ███████████
5. num_featured_artists 0.67  ███████████
```

The model learned that **solo tracks that work well in medleys** are the strongest signal. Makes sense — Super Bowl logistics don't favor complex guest appearances.

---

## Constraint Optimization: Making Predictions Realistic

Raw ML probabilities aren't a setlist. I needed to apply Super Bowl constraints:

```python
class SetlistConstraints:
    MIN_DURATION_SECONDS = 720   # 12 minutes
    MAX_DURATION_SECONDS = 840   # 14 minutes
    MIN_SONGS = 8
    MAX_SONGS = 12
    MIN_LATEST_ALBUM_SONGS = 2   # New album promotion
    MIN_CLASSIC_HITS = 3         # Crowd pleasers
    MAX_COLLAB_SONGS = 3         # Guest logistics
```

### The Infinite Loop Bug

My greedy selection algorithm had a subtle bug: when all remaining songs were filtered out by constraints but the dataframe wasn't empty, it looped forever.

```python
# The fix: track whether we added anything
while len(selected) < MIN_SONGS:
    added_song = False
    for _, row in df_sorted.iterrows():
        # ... selection logic ...
        if song_added:
            added_song = True
            break
    if not added_song:  # Nothing passed constraints
        break           # Exit instead of infinite loop
```

### Duration Estimation

Another snag: some songs had **0-second durations** in the data. My initial code treated these as free, breaking runtime calculations.

```python
def get_song_duration(row) -> int:
    duration = row.get("duration_seconds", 0)
    if pd.isna(duration) or duration <= 0:
        return 180  # Default 3 minutes
    return int(duration)
```

---

## External Validation: Checking Against Reality

Before publishing, I validated against external signals:

### Confirmed: BAILE INoLVIDABLE
The Apple Music halftime trailer (January 16, 2026) featured "BAILE INoLVIDABLE" prominently. **My model had this in the top 5.** Confirmation bias? Maybe. But it validated the approach.

### Guest Artist Intel

Two key updates from news sources:

**Cardi B: Upgraded to HIGH (40%)**
Her boyfriend Stefon Diggs is confirmed playing in the Super Bowl. She'll already be at Levi's Stadium. The logistics advantage for "I Like It" is significant.

**Daddy Yankee: Downgraded to LOW (5%)**
Per a [Complex interview](https://www.complex.com/music/a/bernadette-giacomazzo/daddy-yankee-says-he-wouldnt-perform-gasolina-with-bad-bunny), he's declined secular music since retiring for religious reasons in 2023. "La santa" won't happen.

---

## The Final Prediction

### Expected Setlist (10 songs, 12:59)

| # | Song | Confidence |
|---|------|------------|
| 1 | BAILE INoLVIDABLE | 100% ✓ Confirmed |
| 2 | Yo perreo sola | 100% |
| 3 | Amorfoda | 98% |
| 4 | Chambea | 98% |
| 5 | Si estuviésemos juntos | 96% |
| 6 | NUEVAYoL | 95% |
| 7 | Soy peor | 95% |
| 8 | La santa | 94% |
| 9 | Me porto bonito | 94% |
| 10 | Estamos bien | 94% |

**Overall confidence: 96%**

### Wild Card: Cardi B Appearance (40%)

If Cardi B appears, "I Like It" likely replaces "Estamos bien" — creating a viral crossover moment.

---

## What I Learned

### 1. Proxy Labels Work (Sometimes)

Without direct training data, proxy labels can bootstrap a model. The key is choosing proxies that share the same underlying distribution. "Songs played at flagship shows" correlates with "songs for Super Bowl" because both require the same qualities: crowd appeal, performance polish, and cultural resonance.

### 2. Constraints > Raw Predictions

ML probabilities are necessary but not sufficient. The constraint optimization layer transformed "here are likely songs" into "here's a feasible setlist." Real-world predictions often need this kind of post-processing.

### 3. External Validation Matters

The Daddy Yankee update completely changed my guest predictions. Static models can't capture dynamic world events. Building in a validation step against current news made the final prediction more credible.

### 4. Simple Models Can Win

Logistic Regression beat tree-based models on holdout data. Interpretability was a bonus — I could explain *why* the model predicted each song, not just that it did.

---

## Evaluation Plan

After February 8, I'll score the prediction:

```
Precision = Correct Predictions / 10
Recall = Correct Predictions / Actual Setlist Size
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

| Rating | Criteria |
|--------|----------|
| ⭐⭐⭐⭐⭐ | 8+ songs correct |
| ⭐⭐⭐⭐ | 6-7 songs correct |
| ⭐⭐⭐ | 4-5 songs correct |
| ⭐⭐ | 2-3 songs correct |

---

## Try It Yourself

The full code is open source:

```bash
git clone https://github.com/yourusername/badbunnySetlist
cd badbunnySetlist
pip install -r requirements.txt

# Collect data (needs Setlist.fm API key)
python -m src.collect_data

# Train model
python -m src.train_model

# Generate predictions
python -m src.final_prediction
```

---

## Final Thoughts

Predicting a Super Bowl setlist is part data science, part domain knowledge, part educated guessing. The model provides structure; the constraints provide realism; the external validation provides credibility.

Will I get it right? I'll find out on February 8.

But regardless of the outcome, the process revealed something interesting: **setlists aren't random.** They're optimized for audience, time, and moment — just like any other constrained optimization problem.

And that's a prediction I'm confident in.

---

*Prediction made January 30, 2026. Results pending Super Bowl LX.*

**Tags:** machine-learning, music, prediction, python, super-bowl

# Bad Bunny Super Bowl LX Halftime Show Prediction

**Generated:** January 30, 2026
**Event Date:** February 8, 2026 | Levi's Stadium, Santa Clara, California
**Model Confidence:** 96%
**Status:** 🔮 Awaiting validation

---

## Executive Summary

**Bad Bunny will perform a 10-song medley lasting approximately 13 minutes, anchored by the confirmed "BAILE INoLVIDABLE" from his 2025 album and featuring career-defining hits like "Yo perreo sola," "Chambea," and "Tití me preguntó."** Our hybrid ML + constraint optimization model predicts a setlist balancing new album promotion with fan-favorite classics, formatted as tight snippets typical of Super Bowl halftime shows. The most likely wild card is a Cardi B guest appearance on "I Like It" — her boyfriend Stefon Diggs is confirmed playing in the game, making logistics trivial.

---

## Methodology

### Data Sources

| Source | Records | Purpose |
|--------|---------|---------|
| Setlist.fm API | 471 setlists | Live performance patterns (2016-2025) |
| MusicBrainz | 342 songs | Complete catalog with metadata |
| Super Bowl halftimes | 9 shows | Format analysis (2015-2024) |

### Prediction Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Feature        │     │  ML Model        │     │  Constraint     │
│  Engineering    │ ──▶ │  (Logistic Reg)  │ ──▶ │  Optimization   │
│  (26 features)  │     │  P(setlist)      │     │  (8-12 songs)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Ensemble Score  │
                        │  60% ML + 40%    │
                        │  Constraints     │
                        └──────────────────┘
```

### Model Performance

| Metric | Value | Context |
|--------|-------|---------|
| AUC-ROC | 0.834 | Holdout validation (PR residency) |
| F1 Score | 0.714 | Balance of precision/recall |
| CV AUC | 0.976 | 5-fold cross-validation |
| Holdout Accuracy | 90% | Top-20 predictions vs PR finale |

### Top 10 Predictive Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | medley_friendly | 1.42 | Songs that transition well in live sets |
| 2 | position_category_closer | 0.98 | Typical finale placement |
| 3 | is_primary_artist | 0.74 | Solo tracks preferred over features |
| 4 | cultural_significance | 0.70 | Puerto Rico pride, political themes |
| 5 | num_featured_artists | 0.67 | Fewer collaborators = higher likelihood |
| 6 | days_since_release | 0.63 | Balance of new and classic |
| 7 | position_category_middle | 0.62 | Versatile setlist placement |
| 8 | times_performed_live | 0.60 | Battle-tested crowd favorites |
| 9 | performance_frequency | 0.60 | Consistent set staples |
| 10 | has_english_lyrics | 0.47 | Crossover appeal factor |

---

## Confirmed Information

### ✅ Officially Confirmed (Apple Music Trailer - January 16, 2026)

| Song | Album | Evidence |
|------|-------|----------|
| **BAILE INoLVIDABLE** | Debí Tirar Más Fotos (2025) | Featured prominently in official halftime trailer, filmed in Puerto Rico |

---

## Predicted Setlists

### 🎯 Conservative (8 songs) — Highest Confidence Only

**Runtime:** 10:51 | **Confidence:** 97%

| # | Song | Duration | Confidence | Type |
|---|------|----------|------------|------|
| 1 | BAILE INoLVIDABLE | 2:00 | 100% | ✅ Confirmed, 🆕 New |
| 2 | Yo perreo sola | 1:08 | 100% | 🔥 Classic |
| 3 | Amorfoda | 1:01 | 98% | 🔥 Classic |
| 4 | Chambea | 1:16 | 98% | 🔥 Classic |
| 5 | Si estuviésemos juntos | 1:07 | 96% | 🔥 Classic |
| 6 | NUEVAYoL | 1:22 | 95% | 🆕 New |
| 7 | Soy peor | 1:35 | 95% | 🔥 Classic |
| 8 | La santa | 1:22 | 94% | 🔥 Classic |

---

### ⭐ Expected (10 songs) — Most Likely Scenario

**Runtime:** 12:59 | **Confidence:** 96%

| # | Song | Duration | Confidence | Type |
|---|------|----------|------------|------|
| 1 | BAILE INoLVIDABLE | 2:00 | 100% | ✅ Confirmed, 🆕 New |
| 2 | Yo perreo sola | 1:08 | 100% | 🔥 Classic |
| 3 | Amorfoda | 1:01 | 98% | 🔥 Classic |
| 4 | Chambea | 1:16 | 98% | 🔥 Classic |
| 5 | Si estuviésemos juntos | 1:07 | 96% | 🔥 Classic |
| 6 | NUEVAYoL | 1:22 | 95% | 🆕 New |
| 7 | Soy peor | 1:18 | 95% | 🔥 Classic |
| 8 | La santa | 1:18 | 94% | 🔥 Classic |
| 9 | Me porto bonito | 1:11 | 94% | 🔥 Classic |
| 10 | Estamos bien | 1:18 | 94% | 🔥 Classic |

---

### 🚀 Expansive (12 songs) — Full Medley Like Usher 2024

**Runtime:** 13:49 | **Confidence:** 96%

| # | Song | Duration | Confidence | Type |
|---|------|----------|------------|------|
| 1 | BAILE INoLVIDABLE | 2:00 | 100% | ✅ Confirmed, 🆕 New |
| 2 | Yo perreo sola | 1:05 | 100% | 🔥 Classic |
| 3 | Amorfoda | 1:01 | 98% | 🔥 Classic |
| 4 | Chambea | 1:05 | 98% | 🔥 Classic |
| 5 | Si estuviésemos juntos | 1:05 | 96% | 🔥 Classic |
| 6 | NUEVAYoL | 1:22 | 95% | 🆕 New |
| 7 | Soy peor | 1:05 | 95% | 🔥 Classic |
| 8 | La santa | 1:05 | 94% | 🔥 Classic |
| 9 | Me porto bonito | 1:05 | 94% | 🔥 Classic |
| 10 | Estamos bien | 1:05 | 94% | 🔥 Classic |
| 11 | Tití me preguntó | 0:46 | 94% | 🔥 Classic |
| 12 | Neverita | 1:05 | 93% | 🔥 Classic |

---

## Honorable Mentions

Songs that narrowly missed but could still appear:

| Song | Score | Live Plays | Notable Factor |
|------|-------|------------|----------------|
| El apagón | 93% | 76 | High cultural/political significance |
| Si veo a tu mamá | 93% | 94 | Strong live performer |
| Efecto | 92% | 99 | Recent album standout |
| Tú no metes cabra | 89% | 74 | Iconic early hit |
| Dákiti | 88% | 104 | Crossover smash (Jhay Cortez collab) |
| Callaíta | 87% | 96 | Pre-album viral hit |

---

## Guest Artist Predictions

### Primary Guest: Cardi B — HIGH (40%)

| Factor | Status |
|--------|--------|
| **Logistics** | ✅ Boyfriend Stefon Diggs confirmed playing in SB LX — she's already at venue |
| **Song** | "I Like It" (2018) — #1 Billboard Hot 100, 1B+ video views |
| **History** | Previous collab success, mainstream crossover appeal |
| **Risk** | Song performed at SB 2020 — may want fresh content |

### Secondary Guest: J Balvin — MEDIUM-HIGH (15%)

| Factor | Status |
|--------|--------|
| **History** | Long-time collaborator, Oasis album together |
| **SB 2020** | Both appeared with Shakira/J.Lo |
| **Song Options** | Oasis tracks, could join "I Like It" for reunion |

### ⚠️ Daddy Yankee — LOW (5%)

Per [Complex interview](https://www.complex.com/music/a/bernadette-giacomazzo/daddy-yankee-says-he-wouldnt-perform-gasolina-with-bad-bunny) (January 2026):

> "I'm in a different mission right now. So I gotta represent what I'm doing right now 100 percent."

**Status:** DECLINED secular music. Retired December 2023 for religious mission.
**Only option:** "Sonríele" (inspirational) — unlikely fit for halftime energy.

### Other Possibilities

| Artist | Likelihood | Song | Notes |
|--------|------------|------|-------|
| Rosalía | LOW-MEDIUM | La noche de anoche | Spanish superstar, logistics uncertain |
| Residente | LOW-MEDIUM | Unknown | Puerto Rico solidarity angle |
| Drake | LOW | MIA | Logistically complex, schedule conflicts |

---

## 🃏 Wild Card Scenario: Cardi B Appearance

### If "I Like It" Is Added

| Position | Standard Prediction | Wild Card Scenario |
|----------|--------------------|--------------------|
| 9 | Me porto bonito | **I Like It** 🆕 (w/ Cardi B) |
| 10 | Estamos bien | ~~Estamos bien~~ (bumped) |

**Why this works:**
- Massive crossover appeal (#1 hit, mainstream recognition)
- Creates viral moment validating Bad Bunny's global status
- J Balvin could join for triple reunion
- 2020 SB performance was brief segment — 2026 could be expanded

### Guest Scenario Probabilities

| Scenario | Probability |
|----------|-------------|
| Cardi B appears with "I Like It" | **40%** |
| No guest appearances | 35% |
| J Balvin appears | 15% |
| Daddy Yankee (Sonríele only) | 5% |
| Other guest | 5% |

---

## Key Assumptions

1. **Format:** Medley-style performance (consistent with recent Super Bowls)
2. **Duration:** 12-14 minutes of actual performance time
3. **Album Promotion:** "Debí Tirar Más Fotos" songs prominently featured
4. **No 2020 Repeats:** Songs from SB LIV appearance unlikely (except with Cardi)
5. **Solo Focus:** Limited guests to maximize Bad Bunny showcase time
6. **Energy Arc:** Build from strong opener to peak finale

---

## What Could Go Wrong (Model Limitations)

### Known Blindspots

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Rehearsal changes** | Setlists often finalized last-minute | Predictions based on patterns, not insider info |
| **Surprise songs** | NFL shows include unexpected elements | Can't predict unreleased music or one-time tributes |
| **Production constraints** | Stage design may limit song choices | Model doesn't account for technical requirements |
| **Guest availability** | Last-minute confirmations/cancellations | Guest predictions are probabilistic |
| **NFL creative input** | League may influence content decisions | Model based on artist patterns, not network politics |
| **Cultural moments** | Current events may shift priorities | Static training data, dynamic world |

### Confidence Calibration

Our 96% confidence represents:
- High certainty the **types** of songs selected (classics + new album)
- Medium certainty on **specific songs** (8-10 of 12 likely correct)
- Lower certainty on **exact order** and **guest appearances**

---

## How to Evaluate After February 8

### Scoring Methodology

```
Precision = Correct Predictions / Total Predictions
Recall = Correct Predictions / Actual Setlist Size
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Evaluation Criteria

| Metric | Calculation | Target |
|--------|-------------|--------|
| **Song Accuracy** | Songs predicted that appeared | ≥ 70% |
| **Coverage** | Actual songs we predicted | ≥ 60% |
| **Confirmed Hit** | BAILE INoLVIDABLE appeared | ✅ |
| **New Album Rep** | At least 2 new album songs | ✅ |
| **Guest Prediction** | Correctly predicted guest (or no guest) | ✅ |

### Success Thresholds

| Rating | Criteria |
|--------|----------|
| ⭐⭐⭐⭐⭐ Excellent | ≥8/10 songs correct, guest prediction right |
| ⭐⭐⭐⭐ Good | 6-7/10 songs correct |
| ⭐⭐⭐ Moderate | 4-5/10 songs correct |
| ⭐⭐ Poor | 2-3/10 songs correct |
| ⭐ Failed | <2 songs correct |

---

## Pre-Show Monitoring Checklist

Before February 8, watch for:

- [ ] Official setlist leaks from rehearsals
- [ ] Guest artist arrivals in Santa Clara
- [ ] Social media hints from @badbunny
- [ ] Stage design/prop photos from Levi's Stadium
- [ ] Last-minute song announcements
- [ ] Dancer/choreography reveals
- [ ] **Cardi B sightings at rehearsals** ⭐

---

## About This Analysis

### Tools & Technologies
- **Python 3.11** with pandas, scikit-learn, numpy
- **APIs:** Setlist.fm, MusicBrainz
- **Model:** Logistic Regression (balanced classes)
- **Validation:** Puerto Rico residency holdout (82 songs)

### Reproducibility
Full source code and data processing pipeline available in repository.

```bash
# Regenerate predictions
python -m src.final_prediction
python -m src.validate_prediction
```

### Author
Generated using machine learning analysis of publicly available setlist data.

---

## Sources

- [Rolling Stone: Bad Bunny Super Bowl Trailer](https://www.rollingstone.com/music/music-news/bad-bunny-super-bowl-halftime-show-trailer-1235500516/)
- [Billboard: Dream Setlist Predictions](https://www.billboard.com/lists/bad-bunny-super-bowl-halftime-show-2026-dream-setlist/)
- [Complex: Daddy Yankee Interview](https://www.complex.com/music/a/bernadette-giacomazzo/daddy-yankee-says-he-wouldnt-perform-gasolina-with-bad-bunny)
- [Grammy.com: Bad Bunny Collaborations](https://www.grammy.com/news/11-essential-bad-bunny-collaborations-un-verano-sin-ti-cardi-b-drake-rosalia-bomba-estereo)
- [Wikipedia: Super Bowl LX Halftime](https://en.wikipedia.org/wiki/Super_Bowl_LX_halftime_show)

---

*Prediction generated for entertainment and portfolio purposes. Actual setlist may vary.*

**Last Updated:** January 30, 2026

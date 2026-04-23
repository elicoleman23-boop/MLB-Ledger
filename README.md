# The Hit Ledger — v2

A Python Streamlit application for calculating the probability of MLB batters getting hits in today's games via Monte Carlo simulation. Each plate appearance is modeled separately based on whether the batter is facing the starter (at a specific time through the order) or the bullpen, with pitcher arsenal, park factor, and umpire adjustments applied.

## What's New in v2

v1 treated every PA as "batter vs starter" with one hit probability. v2 models each PA's probability distribution individually:

| PA # | Pitcher | Adjustment |
|---|---|---|
| 1 | Starter | 1st TTO (no penalty) |
| 2 | Starter | 2nd TTO penalty |
| 3 | Starter | 3rd TTO penalty (+larger) |
| 4+ | Bullpen | Team bullpen xBA by handedness |

Plus:

- **Starter workload** — Pitcher's avg IP/start over last 5 starts determines how many PAs the batter sees vs the starter vs the bullpen.
- **TTO penalty** — Pitcher-specific TTO splits when sample is sufficient (≥200 PAs for 2nd TTO, ≥150 for 3rd); flat penalty scaled by pitcher quality otherwise.
- **Bullpen** — Team-level bullpen xBA-against by batter handedness, regressed to league mean (k=500 PAs).
- **Umpire K%** — Home plate umpire's career K% deviation from league average converts to an xBA adjustment (+1pp K% → −0.003 xBA) applied to every PA in the game.
- **BvP annotation** — Optional UI toggle showing lifetime batter-vs-pitcher history. Statistically informational only; does not affect model probabilities.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Click **Run Engine** in the sidebar to pull today's slate.

## What It Does (Full Pipeline)

1. Pull today's schedule + confirmed lineups via MLB StatsAPI
2. For each batter, pull 2 seasons of Statcast pitch-level data
3. For each probable pitcher, pull their season pitch mix
4. For each probable pitcher, compute avg IP over last 5 starts + season xBA-against
5. For each probable pitcher, compute TTO splits (xBA by 1st/2nd/3rd time through)
6. For each team, compute bullpen xBA-against by batter handedness (regressed to league mean)
7. For each game, scrape the home plate umpire + their career K% deviation
8. Build per-PA probability sequences for every batter
9. Run 10,000-game Monte Carlo per batter, sampling each PA from its own distribution
10. Output: P(1+ hit), P(2+ hits), P(1+ HR), P(TB ≥ 1.5), P(TB ≥ 2.5)

## Architecture

```
app.py                              ← entry point
hit_ledger/
├── config.py                       ← all tunable model parameters
├── data/
│   ├── cache.py                    ← SQLite daily snapshot layer (12 tables)
│   ├── lineups.py                  ← MLB StatsAPI
│   ├── statcast.py                 ← pybaseball batter profiles + starter arsenals
│   ├── pitcher_workload.py         ← IP/start + TTO splits (v2)
│   ├── bullpen.py                  ← team bullpen xBA-against (v2)
│   ├── umpires.py                  ← UmpScores + StatsAPI officials (v2)
│   └── bvp.py                      ← batter-vs-pitcher annotation (v2)
├── sim/
│   ├── matchup_v2.py               ← per-PA probability builder
│   ├── engine_v2.py                ← vectorized per-PA Monte Carlo (10k sims)
│   └── pipeline_v2.py              ← end-to-end orchestrator
├── ui/
│   ├── app.py                      ← Streamlit UI
│   └── styles.py                   ← custom CSS
└── utils/odds.py                   ← American odds math (hook, not wired)
```

## Performance

The Monte Carlo engine remains fully vectorized. v2 adds a small per-batter loop to populate the per-PA cumulative probability tensor, but sampling is one big tensor op.

| Batters | 10k sims time (v2) |
|---|---|
| 50 | 0.28s |
| 200 | 0.95s |
| 300 (full slate) | 1.26s |

The bottleneck is Statcast scrapes. First run of the day takes 10–30 minutes (v2 pulls more data than v1 because of TTO/workload/bullpen analyses). Cached runs are instant.

## Caching Strategy

SQLite stores daily snapshots of:
- Games, lineups, batter profiles, pitcher arsenals (v1)
- Starter workload, TTO splits, bullpen profiles, umpire assignments, BvP history (v2)

All queries hit the cache first. Force a full re-scrape via the sidebar's **Advanced → Force refresh** checkbox.

## Model Parameters (all in `config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `REGRESSION_K` | 200 | Pitch-type split regression |
| `SEASON_WEIGHT` / `RECENT_WEIGHT` | 0.70 / 0.30 | Season-long vs last 15 days |
| `PA_BY_LINEUP_SLOT` | 4.5 (top) → 3.7 (bottom) | Total PAs per game |
| `N_SIMULATIONS` | 10,000 | Games per batter |
| `STARTER_RECENT_STARTS` | 5 | IP/start rolling window |
| `BULLPEN_REGRESSION_K` | 500 | Bullpen xBA regression |
| `TTO_MIN_PA_2ND` / `TTO_MIN_PA_3RD` | 200 / 150 | Sample thresholds for pitcher-specific TTO |
| `TTO_FLAT_PENALTY` | 0.000 / 0.010 / 0.020 / 0.030 | Flat fallback by TTO |
| `TTO_QUALITY_MULT_CAP` | 0.5 | Max ±50% scaling on flat penalty |
| `UMPIRE_K_XBA_SENSITIVITY` | 0.003 | xBA subtracted per 1pp K% above league |
| `UMPIRE_K_DEVIATION_CAP` | 0.04 | ±4pp K% max (prevents scrape outliers) |
| `BVP_DEFAULT_ENABLED` | False | Annotation-only, off by default |
| `BVP_MIN_PA_TO_DISPLAY` | 5 | Below this, don't show |

## TTO Penalty Logic (Hybrid)

For each TTO (2nd, 3rd):

1. If the pitcher has ≥ threshold PAs in both 1st TTO AND this TTO, use `pitcher_tto_xba - pitcher_1st_tto_xba` as the penalty.
2. Otherwise, use the flat penalty scaled by pitcher quality:
   ```
   quality_mult = (pitcher_season_xBA - league_avg) / league_avg
   quality_mult = clip(quality_mult, -0.5, 0.5)
   penalty = flat_penalty × (1 + quality_mult)
   ```

So a rookie with no TTO sample who allows a .280 xBA gets a 3rd-TTO penalty of about 0.023 xBA (0.020 × 1.15), while an elite starter at .215 xBA gets 0.017 xBA (0.020 × 0.85).

## Umpire K% Adjustment

If the home plate umpire has a career K% that's 2pp above league average:

```
ump_xba_adj = -0.02 × 100 × 0.003 = -0.006
```

This gets added to every PA's xBA in that game. Capped at ±0.04 K% deviation to prevent extreme scrape values from dominating the model.

## Using the Odds Hook (for later)

`hit_ledger/utils/odds.py` is built and tested but intentionally not wired into the UI. When ready:

```python
from hit_ledger.utils.odds import edge_pct, kelly_fraction

edge = edge_pct(model_prob=0.62, book_odds=-135)   # ~0.046 (4.6 pts of edge)
stake = kelly_fraction(0.62, -135, kelly_mult=0.25) # quarter-Kelly
```

Wire it up by:
1. Add an odds data source (e.g. The Odds API) in `hit_ledger/data/odds_source.py`
2. Merge odds into the predictions DataFrame on `batter_id`
3. Add an "Edge" column + filter to the Hot List

## Known Limitations

- **Lineups must be posted.** MLB confirmed lineups typically lock 1–3 hours before first pitch. Run close to game time.
- **Statcast data lag.** pybaseball hits Baseball Savant which updates overnight. Last night's games aren't in today's profiles.
- **Umpire scraping is fragile.** UmpScorecards site structure can change; if the scrape fails, the pipeline continues with no umpire adjustment (logged as warning).
- **Bullpen as single entity.** v2 doesn't know *which* reliever will face the batter in the 4th PA — just "some reliever from that team's pen." Per-reliever modeling is a future upgrade.
- **TTO independence assumption.** The sim treats PAs as independent Bernoulli trials per TTO bin. Real TTO effects have more complexity (lineup-turn-through-the-order vs raw PA count) but this approximation is standard.
- **BvP annotation only.** Most public research concludes BvP is noise below ~50 PAs. Enabled as annotation so you can see extreme samples without them corrupting the model.
- **Park factors are static.** Refresh `config.PARK_FACTORS_*` annually.
- **League priors are 2024.** Refresh annually.

## Testing

Two test scripts in the project root:

```bash
python test_sim_perf_v2.py      # Confirms <2s target for 10k × 300 batters w/ per-PA probs
python test_integration_v2.py   # End-to-end matchup builder + sim across 3 contrasting scenarios
```

Both run offline without network calls.

## Phases Status

- [x] Phase 1: Environment & library setup
- [x] Phase 2: Data acquisition (lineups, Statcast, arsenals)
- [x] Phase 3: Simulation engine (v2 with per-PA probabilities)
- [x] Phase 4: Streamlit UI (per-PA breakdowns, umpire panel, BvP toggle)
- [x] Phase 5: Optimization & caching (SQLite + @st.cache_data)
- [x] **v2**: Bullpen modeling
- [x] **v2**: Times Through the Order (TTO) penalty
- [x] **v2**: Batter vs Pitcher annotation
- [x] **v2**: Umpire K% adjustment

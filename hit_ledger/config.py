"""
Central configuration for The Hit Ledger.

All tunable model parameters live here so they can be adjusted
without touching the simulation logic.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "cache"
DB_PATH = CACHE_DIR / "hit_ledger.db"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Statcast data windows
# ---------------------------------------------------------------------------
# How many days of "current season" data to pull for each batter.
# Pulling too wide is slow; too narrow misses platoon samples.
SEASONS_LOOKBACK = 2             # seasons of pitch-level data per batter
RECENT_FORM_DAYS = 15            # window for hot/cold weighting
MIN_PITCHES_PER_SPLIT = 15       # below this, fall back entirely to league mean

# ---------------------------------------------------------------------------
# Model parameters - BATTER SPLITS ARE KING
# ---------------------------------------------------------------------------
# Bayesian regression constants for batter-vs-pitch-type splits.
# adjusted = (n * raw + K * league_avg) / (n + K)
#
# Different stats stabilize at different rates, so each gets its own K:
#   - Contact rate stabilizes fastest (~50-75 PAs)
#   - xBA on contact is moderately noisy (~150-200 PAs to stabilize)
#   - HR rate is very noisy and needs heavy regression
REGRESSION_K_CONTACT = 60
REGRESSION_K_XBA = 120
REGRESSION_K_HR = 300

# Alias kept for any code that still references the old single-K name.
REGRESSION_K = REGRESSION_K_XBA

# xBA blend: 70% season-long, 30% last-15-day
SEASON_WEIGHT = 0.70
RECENT_WEIGHT = 0.30

# PAs per game by lineup slot
PA_BY_LINEUP_SLOT = {
    1: 4.5, 2: 4.5,
    3: 4.3, 4: 4.3, 5: 4.3,
    6: 4.0, 7: 4.0,
    8: 3.7, 9: 3.7,
}
DEFAULT_PA = 4.2  # fallback if lineup slot unknown

# Monte Carlo
N_SIMULATIONS = 10_000
RNG_SEED = None  # set an int for reproducibility during debugging

# Pitch-by-pitch simulator (Fix E Phase 3). When True, the pipeline swaps
# engine_v2.simulate_v2 for pitch_sim_engine.simulate_pbp. The pbp engine
# is ~20-50x slower than the fast path because every pitch walks through
# Python; default n_sims should drop to ~1000 to keep a slate under ~30s.
USE_PITCH_BY_PITCH_SIM = False
PBP_DEFAULT_N_SIMS = 1_000

# Game-level multiplicative noise on non-HR hit probabilities — one draw
# per simulated game, shared across all PAs. Models persistent within-game
# effects (weather, umpire, defensive form). Centered at 1.0 so means are
# preserved; variance/dispersion grows, producing realistic fat tails for
# team totals and alt-line markets. Set to 0 for deterministic runs.
BABIP_NOISE_SD = 0.12  # 8% relative noise on non-HR hit prob

# ---------------------------------------------------------------------------
# League-average priors (used for shrinkage)
# Sourced from Baseball Savant 2024 league-wide splits; refresh annually.
# ---------------------------------------------------------------------------
LEAGUE_AVG_XBA = 0.244              # xBA on contact (Statcast)
LEAGUE_AVG_ISO = 0.157
LEAGUE_AVG_HR_PER_BBE = 0.043       # HR per batted-ball event
LEAGUE_AVG_K_RATE = 0.225           # Strikeout rate per PA
LEAGUE_AVG_BB_RATE = 0.085          # Walk rate per PA
LEAGUE_AVG_CONTACT_RATE = 0.765     # 1 - K_rate - other (balls in play / PA)

# League HR per batted-ball event by pitch type (2024 Baseball Savant).
# These are HR / (balls in play), NOT HR/PA. Used as regression priors for
# per-pitch-type HR modeling and as the league baseline in log-5 blending.
LEAGUE_HR_PER_CONTACT_BY_PITCH = {
    "FF": 0.055,  # 4-Seam - most homered pitch, especially elevated
    "SI": 0.038,  # Sinker - lower launch angle, fewer HRs
    "FC": 0.045,  # Cutter
    "SL": 0.040,  # Slider
    "ST": 0.035,  # Sweeper
    "CU": 0.040,  # Curveball
    "KC": 0.038,  # Knuckle Curve
    "CH": 0.052,  # Changeup - flat ones get crushed
    "FS": 0.040,  # Splitter
    "FO": 0.035,  # Forkball
    "SC": 0.040,  # Screwball
    "EP": 0.050,  # Eephus
}
LEAGUE_AVG_HR_PER_CONTACT = 0.043   # ties to existing LEAGUE_AVG_HR_PER_BBE

# ---------------------------------------------------------------------------
# Pitch-by-pitch sim (Fix E) — league priors for swing/contact rates and
# batted-ball EV/LA distributions. Used when batter or pitcher has too few
# samples in a (pitch_type, zone) split to trust the observed rate.
# Values are public Baseball Savant splits for 2024; refresh annually.
# ---------------------------------------------------------------------------

# Minimum PA-ending contact events needed to trust a per-pitch-type EV/LA
# distribution. Below this, fall back to the batter's overall EV/LA, then
# to LEAGUE_EV_LA.
MIN_EV_LA_SAMPLE = 40

# League-wide batted-ball distribution (all pitch types pooled). Used as the
# ultimate fallback when per-pitch and overall samples are both too sparse.
# Calibrated to 2024 MLB league averages. Refresh annually or when adding a
# pybaseball-based calibration harness (Gap-2 follow-up).
LEAGUE_EV_LA = {
    "mean_ev": 89.1,   # mph
    "sd_ev": 14.8,
    "mean_la": 11.5,   # degrees
    "sd_la": 28.2,
    "corr_ev_la": -0.08,
}

# P(swing | in-zone) per pitch type.
# Calibrated to 2024 MLB league averages. Refresh annually or when adding a
# pybaseball-based calibration harness (Gap-2 follow-up).
LEAGUE_Z_SWING_BY_PITCH = {
    "FF": 0.69, "SI": 0.73, "FC": 0.68, "SL": 0.62, "ST": 0.60,
    "CU": 0.55, "KC": 0.58, "CH": 0.64, "FS": 0.62, "FO": 0.56,
    "SC": 0.60, "EP": 0.52,
}

# P(swing | out-of-zone) per pitch type.
# Calibrated to 2024 MLB league averages. Refresh annually or when adding a
# pybaseball-based calibration harness (Gap-2 follow-up).
LEAGUE_O_SWING_BY_PITCH = {
    "FF": 0.18, "SI": 0.23, "FC": 0.26, "SL": 0.33, "ST": 0.31,
    "CU": 0.25, "KC": 0.27, "CH": 0.31, "FS": 0.35, "FO": 0.31,
    "SC": 0.27, "EP": 0.12,
}

# P(contact | swing, in-zone) per pitch type.
# Calibrated to 2024 MLB league averages. Refresh annually or when adding a
# pybaseball-based calibration harness (Gap-2 follow-up).
LEAGUE_Z_CONTACT_BY_PITCH = {
    "FF": 0.89, "SI": 0.91, "FC": 0.86, "SL": 0.80, "ST": 0.77,
    "CU": 0.83, "KC": 0.81, "CH": 0.83, "FS": 0.79, "FO": 0.79,
    "SC": 0.83, "EP": 0.87,
}

# P(contact | swing, out-of-zone) per pitch type.
# Calibrated to 2024 MLB league averages. Refresh annually or when adding a
# pybaseball-based calibration harness (Gap-2 follow-up).
LEAGUE_O_CONTACT_BY_PITCH = {
    "FF": 0.64, "SI": 0.66, "FC": 0.58, "SL": 0.48, "ST": 0.43,
    "CU": 0.55, "KC": 0.52, "CH": 0.56, "FS": 0.50, "FO": 0.49,
    "SC": 0.55, "EP": 0.60,
}

# Per-pitch-type league averages for xBA AND whiff rates
# These are critical for realistic simulation
LEAGUE_XBA_BY_PITCH = {
    "FF": 0.260,  # 4-Seam Fastball
    "SI": 0.270,  # Sinker
    "FC": 0.235,  # Cutter
    "SL": 0.215,  # Slider
    "ST": 0.210,  # Sweeper
    "CU": 0.220,  # Curveball
    "KC": 0.215,  # Knuckle Curve
    "CH": 0.235,  # Changeup
    "FS": 0.220,  # Splitter
    "FO": 0.210,  # Forkball
    "SC": 0.230,  # Screwball
    "EP": 0.230,  # Eephus
}

# Whiff rates by pitch type (how often batters swing and miss)
# Higher whiff = lower contact = lower hit probability
LEAGUE_WHIFF_BY_PITCH = {
    "FF": 0.22,   # 4-Seam - lower whiff, easier to hit
    "SI": 0.18,   # Sinker - low whiff, induces grounders
    "FC": 0.25,   # Cutter - moderate
    "SL": 0.35,   # Slider - high whiff pitch
    "ST": 0.38,   # Sweeper - very high whiff
    "CU": 0.32,   # Curveball - high whiff
    "KC": 0.34,   # Knuckle Curve - high whiff
    "CH": 0.33,   # Changeup - high whiff
    "FS": 0.36,   # Splitter - very high whiff
    "FO": 0.34,   # Forkball - high whiff
    "SC": 0.28,   # Screwball
    "EP": 0.15,   # Eephus - low whiff but rare
}

# ---------------------------------------------------------------------------
# Park factors — simple 1.00-centered multipliers on hit probability
# Values > 1.0 favor hitters. Sourced from ballpark factor averages (2022-2024).
# Refine annually; these are reasonable v1 defaults.
# ---------------------------------------------------------------------------
PARK_FACTORS_HITS = {
    "Coors Field": 1.12,
    "Fenway Park": 1.06,
    "Globe Life Field": 1.04,
    "Great American Ball Park": 1.04,
    "Wrigley Field": 1.03,
    "Yankee Stadium": 1.02,
    "Citizens Bank Park": 1.02,
    "Chase Field": 1.02,
    "Minute Maid Park": 1.01,
    "Kauffman Stadium": 1.01,
    "Rogers Centre": 1.01,
    "Target Field": 1.00,
    "American Family Field": 1.00,
    "Busch Stadium": 1.00,
    "Nationals Park": 1.00,
    "loanDepot park": 0.99,
    "Citi Field": 0.99,
    "PNC Park": 0.99,
    "Truist Park": 0.99,
    "Angel Stadium": 0.98,
    "Dodger Stadium": 0.98,
    "Progressive Field": 0.98,
    "Guaranteed Rate Field": 0.98,
    "Comerica Park": 0.97,
    "Oracle Park": 0.96,
    "T-Mobile Park": 0.96,
    "Petco Park": 0.96,
    "Oakland Coliseum": 0.95,
    "Tropicana Field": 0.95,
    "Camden Yards": 0.97,
    # Default for any unlisted park
    "_default": 1.00,
}

PARK_FACTORS_HR = {
    "Coors Field": 1.15,
    "Great American Ball Park": 1.20,
    "Yankee Stadium": 1.18,
    "Globe Life Field": 1.08,
    "Citizens Bank Park": 1.10,
    "Wrigley Field": 1.05,
    "Fenway Park": 1.00,  # big wall suppresses some HR
    "Chase Field": 1.05,
    "Minute Maid Park": 1.04,
    "Rogers Centre": 1.04,
    "Angel Stadium": 0.98,
    "Dodger Stadium": 1.02,
    "Petco Park": 0.92,
    "Oracle Park": 0.85,
    "T-Mobile Park": 0.90,
    "Tropicana Field": 0.95,
    "Oakland Coliseum": 0.88,
    "Comerica Park": 0.93,
    "Kauffman Stadium": 0.95,
    "_default": 1.00,
}

# ---------------------------------------------------------------------------
# Outcome distribution given a hit (league-wide, for simulation)
# ---------------------------------------------------------------------------
# P(1B | hit), P(2B | hit), P(3B | hit), P(HR | hit)
# Derived from 2024 league totals. HR share is batter-specific; this is the fallback.
HIT_TYPE_DIST = {
    "1B": 0.652,
    "2B": 0.195,
    "3B": 0.018,
    "HR": 0.135,
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HOT_LIST_SIZE = 5

# ===========================================================================
# v2 additions: bullpen, TTO, umpire, BvP
# ===========================================================================

# ---------------------------------------------------------------------------
# Starter workload — controls expected starter vs bullpen PA split
# ---------------------------------------------------------------------------
STARTER_RECENT_STARTS = 5               # IP/start rolling window
LEAGUE_AVG_STARTER_IP = 5.3             # fallback if pitcher has <2 starts
LEAGUE_AVG_PA_PER_INNING = 4.3          # PAs per inning leaguewide (~4.25-4.35)

# ---------------------------------------------------------------------------
# Bullpen — team-level xBA-against by handedness
# ---------------------------------------------------------------------------
LEAGUE_AVG_BULLPEN_XBA_VS_R = 0.243
LEAGUE_AVG_BULLPEN_XBA_VS_L = 0.248
BULLPEN_REGRESSION_K = 250              # PA threshold for bullpen regression

# ---------------------------------------------------------------------------
# Times Through the Order (TTO)
# ---------------------------------------------------------------------------
# Pitcher-specific TTO only trusted above these sample thresholds.
# Below threshold, fall back to league flat penalty scaled by pitcher quality.
TTO_MIN_PA_2ND = 200
TTO_MIN_PA_3RD = 150

# Flat league penalties (xBA delta vs 1st TTO)
TTO_FLAT_PENALTY = {
    1: 0.000,
    2: 0.010,
    3: 0.020,
    4: 0.030,   # rare, but happens with elite starters
}

# Quality scaling: worse-than-league starters get TTO_FLAT_PENALTY × (1 + quality_mult)
# where quality_mult = (pitcher_xBA_against - league_avg) / league_avg, capped ±0.5
TTO_QUALITY_MULT_CAP = 0.5

# ---------------------------------------------------------------------------
# Umpire K% adjustment
# ---------------------------------------------------------------------------
# +1 pp K% above league average → this much subtracted from xBA per PA
UMPIRE_K_XBA_SENSITIVITY = 0.003
LEAGUE_AVG_UMPIRE_K_PCT = 0.226         # 2024 league K%
UMPIRE_K_DEVIATION_CAP = 0.04           # ±4 pp K deviation max (prevents extreme scrapes)

# ---------------------------------------------------------------------------
# BvP — enabled by default for annotations
# ---------------------------------------------------------------------------
BVP_DEFAULT_ENABLED = True
BVP_MIN_PA_TO_DISPLAY = 5               # hide tiny samples from UI

# ---------------------------------------------------------------------------
# Umpire scrape targets (documented; used by data.umpires)
# ---------------------------------------------------------------------------
UMPSCORES_BASE_URL = "https://umpscorecards.com"


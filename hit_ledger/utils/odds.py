"""
American odds math — the hook for future book-line integration.

Left deliberately un-wired in the UI per user spec. When ready to compare
model probabilities to live book lines, add a data source (e.g. The Odds API
or OddsJam) that returns a DataFrame of batter × market × odds, then merge
on batter_id and call `edge_pct` on each row.
"""
from __future__ import annotations

import math


def american_to_implied_prob(odds: int) -> float:
    """
    Convert American moneyline odds to implied probability (no-vig NOT removed).

    Examples:
        -150  ->  0.600
        +130  ->  0.435
    """
    if odds == 0:
        raise ValueError("odds cannot be 0")
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def implied_prob_to_american(prob: float) -> int:
    """
    Convert a probability to American odds.

    Examples:
        0.60  -> -150
        0.40  -> +150
    """
    if not (0 < prob < 1):
        raise ValueError(f"prob must be in (0, 1); got {prob}")
    if prob >= 0.5:
        return -int(round(100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def edge_pct(model_prob: float, book_odds: int) -> float:
    """
    Return the edge in percentage points: model_prob - implied_prob.

    Positive = model thinks it's more likely than the book.
    """
    implied = american_to_implied_prob(book_odds)
    return model_prob - implied


def kelly_fraction(model_prob: float, book_odds: int, kelly_mult: float = 0.25) -> float:
    """
    Return the fraction of bankroll to wager under fractional Kelly.

    Defaults to quarter-Kelly which is standard for sports betting where
    the true edge is uncertain.
    """
    if book_odds > 0:
        b = book_odds / 100
    else:
        b = 100 / (-book_odds)

    p = model_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    return max(0.0, full_kelly * kelly_mult)


def no_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    """
    Strip the vig from a two-way market (e.g., Over/Under).

    Returns (true_prob_a, true_prob_b) that sum to 1.0.
    """
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    total = p_a + p_b
    if total == 0:
        raise ValueError("invalid odds")
    return p_a / total, p_b / total


def fmt_american(odds: int) -> str:
    """Format American odds with explicit sign, e.g. '+135' or '-150'."""
    return f"{odds:+d}"

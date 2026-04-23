"""Team name mappings shared across pipeline and UI."""
from __future__ import annotations

TEAM_SHORT_TO_FULL: dict[str, str] = {
    "D-backs": "Arizona Diamondbacks", "Diamondbacks": "Arizona Diamondbacks",
    "Braves": "Atlanta Braves", "Orioles": "Baltimore Orioles",
    "Red Sox": "Boston Red Sox", "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox", "Reds": "Cincinnati Reds",
    "Guardians": "Cleveland Guardians", "Rockies": "Colorado Rockies",
    "Tigers": "Detroit Tigers", "Astros": "Houston Astros",
    "Royals": "Kansas City Royals", "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers", "Marlins": "Miami Marlins",
    "Brewers": "Milwaukee Brewers", "Twins": "Minnesota Twins",
    "Mets": "New York Mets", "Yankees": "New York Yankees",
    "Athletics": "Oakland Athletics", "Phillies": "Philadelphia Phillies",
    "Pirates": "Pittsburgh Pirates", "Padres": "San Diego Padres",
    "Mariners": "Seattle Mariners", "Giants": "San Francisco Giants",
    "Cardinals": "St. Louis Cardinals", "Rays": "Tampa Bay Rays",
    "Rangers": "Texas Rangers", "Blue Jays": "Toronto Blue Jays",
    "Nationals": "Washington Nationals",
}

TEAM_FULL_TO_SHORT: dict[str, str] = {v: k for k, v in TEAM_SHORT_TO_FULL.items()}

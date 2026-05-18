"""
Kelly criterion stake sizing for sports betting.
Base unit = $10. Uses fractional Kelly (25%) to size bets in units.
"""

UNIT = 10.0  # $10 per unit


def kelly_units(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.15,
    max_units: float = 5.0,
) -> float:
    """
    Returns recommended bet size in units (1 unit = $10).
    Fractional Kelly, capped at max_units per bet.
    Returns 0 if no edge.
    """
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b
    if full_kelly <= 0:
        return 0.0
    units = full_kelly * fraction * 100   # treat 100 units as the "bankroll"
    return round(min(units, max_units), 2)


def kelly_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Dollar stake given a bankroll (legacy, used by backtest)."""
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b
    if full_kelly <= 0:
        return 0.0
    stake = bankroll * full_kelly * fraction
    return round(min(stake, bankroll * 0.10), 2)


def expected_value(model_prob: float, decimal_odds: float) -> float:
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def to_decimal(american: int) -> float:
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def bet_summary(model_prob: float, odds_american: int, nv_prob: float,
                bankroll: float = 1000.0, fraction: float = 0.15) -> dict:
    dec = to_decimal(odds_american)
    edge = model_prob - nv_prob
    units = kelly_units(model_prob, dec, fraction)
    dollars = round(units * UNIT, 2)
    ev = expected_value(model_prob, dec)
    return {
        "model_prob": round(model_prob, 4),
        "market_nv_prob": round(nv_prob, 4),
        "edge": round(edge, 4),
        "decimal_odds": round(dec, 3),
        "american_odds": odds_american,
        "ev": round(ev, 4),
        "units": units,
        "dollars": dollars,
        "has_edge": edge > 0,
        "kelly_fraction": fraction,
    }

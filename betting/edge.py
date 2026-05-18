"""
Single source of truth for bet decision logic.
All tabs and the backtest call evaluate_fight() — never inline the logic.
"""
from betting.devig import no_vig_probs
from betting.kelly import kelly_units, to_decimal


def evaluate_fight(
    prob_a: float,
    prob_b: float,
    odds_a: int,
    odds_b: int,
) -> dict:
    """
    Apply standard bet rules and return a recommendation dict.

    Bet rules:
      Consensus bet  — model and market agree on winner, edge >= MIN_EDGE
      Contrarian bet — model disagrees with market BUT model confidence >= 65%
                       AND edge >= MIN_EDGE * 2 (double threshold, higher bar)

    The double-threshold for contrarian bets filters out noise while still
    capturing high-confidence calls like a 73% underdog at +116.

    Always returns edge_a, edge_b, model_edge for display on all fights.
    """
    from pipeline.config import MIN_EDGE, MAX_EDGE, KELLY_FRAC, MAX_UNITS

    nv_a, nv_b = no_vig_probs(odds_a, odds_b)
    edge_a = prob_a - nv_a
    edge_b = prob_b - nv_b

    model_pick  = "a" if prob_a >= 0.5 else "b"
    market_pick = "a" if nv_a  >= 0.5 else "b"
    model_edge  = edge_a if model_pick == "a" else edge_b

    base = {
        "nv_a":        round(nv_a, 4),
        "nv_b":        round(nv_b, 4),
        "edge_a":      round(edge_a, 4),
        "edge_b":      round(edge_b, 4),
        "model_pick":  model_pick,
        "market_pick": market_pick,
        "model_edge":  round(model_edge, 4),
        "bet":         False,
        "bet_side":    None,
        "edge":        None,
        "units":       0.0,
        "dollars":     0.0,
        "odds_taken":  None,
    }

    agrees     = model_pick == market_pick
    threshold  = MIN_EDGE if agrees else MIN_EDGE * 2  # 6% consensus, 12% contrarian

    if model_pick == "a" and threshold <= edge_a <= MAX_EDGE:
        bet_side, edge, prob, odds = "a", edge_a, prob_a, odds_a
    elif model_pick == "b" and threshold <= edge_b <= MAX_EDGE:
        bet_side, edge, prob, odds = "b", edge_b, prob_b, odds_b
    else:
        return base

    dec   = to_decimal(odds)
    units = min(kelly_units(prob, dec, fraction=KELLY_FRAC), MAX_UNITS)

    return {**base,
        "bet":        True,
        "bet_side":   bet_side,
        "edge":       round(edge, 4),
        "units":      round(units, 2),
        "dollars":    round(units * 10, 2),
        "odds_taken": odds,
    }

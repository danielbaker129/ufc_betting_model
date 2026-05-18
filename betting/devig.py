"""
Devig utilities: convert raw market odds to no-vig true probabilities.
Implements Shin method (best for favorite-longshot bias correction),
multiplicative method, and power method.
"""
import math


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def implied_prob(american_odds: int) -> float:
    dec = american_to_decimal(american_odds)
    return 1.0 / dec


def devig_multiplicative(odds_a: int, odds_b: int) -> tuple[float, float]:
    pa = implied_prob(odds_a)
    pb = implied_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def devig_power(odds_a: int, odds_b: int) -> tuple[float, float]:
    """Power method: find k such that pa^k + pb^k = 1."""
    pa_raw = implied_prob(odds_a)
    pb_raw = implied_prob(odds_b)

    def total_at_k(k):
        return pa_raw ** k + pb_raw ** k

    k_lo, k_hi = 0.5, 2.0
    for _ in range(50):
        k_mid = (k_lo + k_hi) / 2
        if total_at_k(k_mid) > 1.0:
            k_lo = k_mid
        else:
            k_hi = k_mid
    k = (k_lo + k_hi) / 2
    pa = pa_raw ** k
    pb = pb_raw ** k
    return pa / (pa + pb), pb / (pa + pb)


def devig_shin(odds_a: int, odds_b: int, tol: float = 1e-10, max_iter: int = 1000) -> tuple[float, float]:
    """
    Shin method: iterative solution for no-vig probabilities.
    Best method for two-outcome markets; accounts for favorite-longshot bias.
    Returns (prob_a, prob_b) where prob_a + prob_b = 1.0
    """
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    q_a = 1.0 / dec_a
    q_b = 1.0 / dec_b
    overround = q_a + q_b

    if abs(overround - 1.0) < 1e-6:
        return q_a, q_b

    # Shin iterative
    z = 0.0
    for _ in range(max_iter):
        denom_a = math.sqrt(z**2 + 4 * (1 - z) * q_a**2 / overround)
        denom_b = math.sqrt(z**2 + 4 * (1 - z) * q_b**2 / overround)
        p_a = (denom_a - z) / (2 * (1 - z))
        p_b = (denom_b - z) / (2 * (1 - z))
        z_new = overround - 1.0 - (p_a + p_b - 1.0)
        if abs(z_new - z) < tol:
            break
        z = max(0.0, min(z_new, 0.5))

    total = p_a + p_b
    return p_a / total, p_b / total


def no_vig_probs(odds_a: int, odds_b: int, method: str = "shin") -> tuple[float, float]:
    if method == "shin":
        return devig_shin(odds_a, odds_b)
    elif method == "power":
        return devig_power(odds_a, odds_b)
    else:
        return devig_multiplicative(odds_a, odds_b)


def edge(model_prob: float, odds_american: int) -> float:
    """Return edge = model_prob - no_vig_prob for this side."""
    dec = american_to_decimal(odds_american)
    opp_dec = american_to_decimal(-odds_american) if odds_american != 0 else 2.0
    nv_a, _ = no_vig_probs(odds_american, decimal_to_american(opp_dec))
    return model_prob - nv_a


if __name__ == "__main__":
    # Test
    print("Devig test: -150 / +130")
    pa, pb = devig_shin(-150, 130)
    print(f"  Shin:           {pa:.4f} / {pb:.4f} (sum={pa+pb:.4f})")
    pa, pb = devig_multiplicative(-150, 130)
    print(f"  Multiplicative: {pa:.4f} / {pb:.4f}")
    pa, pb = devig_power(-150, 130)
    print(f"  Power:          {pa:.4f} / {pb:.4f}")

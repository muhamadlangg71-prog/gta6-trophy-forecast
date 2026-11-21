"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of Exophase-tracked players who will have earned it at the case's
target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball
loss on the logit scale.

Model (Python standard library only, fully deterministic). The key idea:
earn rates saturate at 100%, so a logit-linear drift overshoots. Work
instead with u = log(1 - p), the log share of players who have NOT earned
the trophy. Under a constant per-player hazard, u falls linearly; the
forecast is a linear extrapolation of u, which saturates gracefully in
rate space.

  * u = log(1 - p), p clipped to [0.05, 99.95] percent.
  * Frozen reads (players_tracked and every rate identical to the previous
    read) carry no information and are collapsed out.
  * Each interval's u move is split into mechanical dilution (new tracked
    players arrive as non-earners) and real progression, using
    players_tracked. Drift is estimated from the progression-adjusted
    series only.
  * Per-trophy drift from a recency-weighted least-squares slope of the
    progression-adjusted u, shrunk toward the mean drift of its tier
    (data-driven bands of the last rate) with an empirical-Bayes weight,
    so flat or short history falls back toward the tier mean.
  * Future players_tracked is forecast with damped log-growth, and the
    future dilution it implies is applied.
  * Per-trophy spread from the observed variance of progression-adjusted
    u increments (floored, so flat history stays honestly uncertain).

Fallback: any trophy (or case) that fails inside the model falls back to
the house baseline (persistence + 0.5/sqrt-day random-walk spread on the
logit scale), so the output is always a valid file.

I/O contract: read every <id>.in in the input directory, write <id>.out to
the output directory:
  {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
one row per trophy, each id once, five finite numbers 0..100, non-decreasing.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The house baseline's spread, in logit units per square-root day (fallback).
BASE_SIGMA = 0.5

# Model constants.
P_LO = 0.05      # percent; lower clip for rates
P_HI = 99.95     # percent; upper clip for rates
OUT_LO = 0.01    # percent; output clamp
OUT_HI = 99.99   # percent; output clamp
RECENCY_HALF_LIFE_DAYS = 1.0   # recency weight half-life for drift fitting
TAU_N_DAYS = 2.0               # damping time for players_tracked growth
DRIFT_CLAMP_U = (-3.0, 0.1)    # u-slope, per day (u falls as trophies are earned)
SIGMA_U_FLOOR = 0.10
SIGMA_U_CAP = 0.60
SIGMA_U_FLAT = 0.25            # used when a trophy shows no movement at all
TAU_FLOOR = 0.05               # floor for the tier spread of true drifts


def clamp_pct(p: float) -> float:
    return min(P_HI, max(P_LO, p))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def u_of_pct(pct: float) -> float:
    """u = log(1 - p), the log share of non-earners. Falls as trophies are earned."""
    return math.log(1.0 - clamp_pct(pct) / 100.0)


def pct_of_u(u: float) -> float:
    """Rate (percent) from u, clipped to the model's range."""
    p = 1.0 - math.exp(u)
    return min(P_HI, max(P_LO, p * 100.0))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    """ISO timestamp -> seconds since epoch (float). None when unparsable."""
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls_slope(ts, ys, weights):
    """Weighted least-squares slope of ys on ts. 0.0 when undefined."""
    sw = sum(weights)
    if sw <= 0:
        return 0.0
    tbar = sum(w * t for w, t in zip(weights, ts)) / sw
    ybar = sum(w * y for w, y in zip(weights, ys)) / sw
    num = sum(w * (t - tbar) * (y - ybar) for w, t, y in zip(weights, ts, ys))
    den = sum(w * (t - tbar) ** 2 for w, t in zip(weights, ts))
    if den <= 1e-12:
        return 0.0
    return num / den


def baseline_quantiles(last_rate_pct, horizon_days):
    """House baseline: persistence + random-walk spread (fallback)."""
    lg = logit_pct(last_rate_pct if last_rate_pct is not None else 0.5)
    spread = BASE_SIGMA * math.sqrt(max(horizon_days, 0.0))
    qs = []
    for z in Z:
        q = 100.0 / (1.0 + math.exp(-(lg + z * spread)))
        qs.append(round4(min(OUT_HI, max(OUT_LO, q))))
    return enforce_monotone(qs)


def enforce_monotone(qs):
    out = []
    prev = None
    for q in qs:
        q = min(OUT_HI, max(OUT_LO, float(q)))
        if prev is not None and q < prev:
            q = prev
        out.append(round4(q))
        prev = out[-1]
    return out


def tier_band(last_rate_pct):
    """Data-driven tier from the last rate (percent)."""
    if last_rate_pct is None:
        return 1
    if last_rate_pct < 1.0:
        return 0
    if last_rate_pct < 10.0:
        return 1
    if last_rate_pct < 50.0:
        return 2
    return 3


def trophy_rows(case, j):
    """Reads for trophy j with a rate: list of (t_days, players, rate_pct).

    Consecutive reads whose full state (players_tracked and every rate) is
    identical to the previous read are collapsed out: a frozen page carries
    no information. t_days is days since the first read; when timestamps
    are missing, read order is used with 30 minutes per slot.
    """
    history = case["history"]
    t0 = None
    stamps = []
    for read in history:
        ts = parse_time(read.get("at"))
        stamps.append(ts)
        if ts is not None and t0 is None:
            t0 = ts
    rows = []
    prev_key = None
    for i, read in enumerate(history):
        rates = read.get("rates") or []
        rate = rates[j] if j < len(rates) else None
        if rate is None:
            continue
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        ts = stamps[i]
        if ts is not None and t0 is not None:
            t = (ts - t0) / 86400.0
        else:
            t = i / 48.0
        n = read.get("players_tracked") or 0
        rows.append((t, float(n), float(rate)))
    return rows


def progression_u_series(rows):
    """Dilution-adjusted u series: list of (t_days, v).

    v accumulates each interval's u move minus the mechanical dilution
    shift implied by players_tracked growth, so v measures real
    progression only.
    """
    if not rows:
        return []
    vs = [(rows[0][0], u_of_pct(rows[0][2]))]
    v = vs[0][1]
    for (t0, n0, p0), (t1, n1, p1) in zip(rows, rows[1:]):
        u0 = u_of_pct(p0)
        u1 = u_of_pct(p1)
        if n1 > n0 > 0:
            # No progression, earners fixed: rate is diluted to p0 * n0/n1.
            diluted_u = math.log(1.0 - (p0 / 100.0) * (n0 / n1))
            dil = diluted_u - u0
        else:
            dil = 0.0
        v += (u1 - u0 - dil)
        vs.append((t1, v))
    return vs


def time_weights(ts):
    """Per-read weights: interval length around the read, recency-weighted."""
    n = len(ts)
    if n == 0:
        return []
    t_last = ts[-1]
    weights = []
    for i, t in enumerate(ts):
        if n == 1:
            w = 1.0
        elif i == 0:
            w = (ts[1] - ts[0]) / 2.0
        elif i == n - 1:
            w = (ts[-1] - ts[-2]) / 2.0
        else:
            w = (ts[i + 1] - ts[i - 1]) / 2.0
        w *= math.exp(-(t_last - t) * math.log(2) / RECENCY_HALF_LIFE_DAYS)
        weights.append(max(w, 0.0))
    return weights


def raw_u_slope(rows):
    """Raw per-trophy u-slope (per day) from the progression-adjusted series."""
    vs = progression_u_series(rows)
    if len(vs) < 2:
        return 0.0
    return wls_slope([t for t, _ in vs], [y for _, y in vs],
                     time_weights([t for t, _ in vs]))


def sigma_u(rows):
    """Spread of the progression-adjusted u increments, per sqrt-day."""
    vs = progression_u_series(rows)
    num = 0.0
    den = 0.0
    moved = False
    for (t0, y0), (t1, y1) in zip(vs, vs[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        dy = y1 - y0
        if abs(dy) > 1e-9:
            moved = True
        num += dy * dy
        den += dt
    if not moved or den <= 0:
        return SIGMA_U_FLAT
    return min(SIGMA_U_CAP, max(SIGMA_U_FLOOR, math.sqrt(num / den)))


def players_growth_rate(case, horizon_days):
    """Forecast players_tracked at the target (damped log-growth)."""
    history = case["history"]
    pts = []
    t0 = None
    for read in history:
        n = read.get("players_tracked") or 0
        if n <= 0:
            continue
        ts = parse_time(read.get("at"))
        if ts is not None:
            if t0 is None:
                t0 = ts
            pts.append(((ts - t0) / 86400.0, math.log(n)))
    if not pts:
        return None
    t_last, logn_last = pts[-1]
    if len(pts) < 2:
        return math.exp(logn_last)
    weights = [math.exp(-(t_last - t) * math.log(2) / RECENCY_HALF_LIFE_DAYS)
               for t, _ in pts]
    g = wls_slope([t for t, _ in pts], [v for _, v in pts], weights)
    g = min(1.0, max(0.0, g))
    damped = g * TAU_N_DAYS * (1.0 - math.exp(-horizon_days / TAU_N_DAYS))
    return math.exp(logn_last + damped)


def tier_statistics(case):
    """(tier_mean_u_slope, tier_tau) per band, from raw per-trophy slopes."""
    trophies = case["trophies"]
    rows_list = [trophy_rows(case, j) for j in range(len(trophies))]
    raw = [raw_u_slope(rows) for rows in rows_list]
    last_rates = [rows[-1][2] if rows else None for rows in rows_list]
    global_mean = sum(raw) / len(raw) if raw else 0.0
    global_std = 0.0
    if len(raw) > 1:
        global_std = math.sqrt(sum((r - global_mean) ** 2 for r in raw) / len(raw))
    stats = []
    for band in range(4):
        members = [raw[j] for j in range(len(trophies))
                   if tier_band(last_rates[j]) == band]
        if len(members) >= 2:
            mean = sum(members) / len(members)
            tau = math.sqrt(sum((r - mean) ** 2 for r in members) / len(members))
        else:
            mean = global_mean
            tau = global_std
        stats.append((mean, max(tau, TAU_FLOOR)))
    return stats


def forecast_trophy(case, j, horizon_days, tier_stats, n_hat):
    """Five quantiles for trophy j. Falls back to the baseline on any issue."""
    rows = trophy_rows(case, j)
    last_rate = rows[-1][2] if rows else None
    last_n = rows[-1][1] if rows else 0
    if not rows:
        return baseline_quantiles(0.5, horizon_days)
    try:
        vs = progression_u_series(rows)
        t_last = vs[-1][0]
        span = max(t_last - vs[0][0], 0.0)
        raw_slope = raw_u_slope(rows)
        sig = sigma_u(rows)
        band = tier_band(last_rate)
        tier_mean, tier_tau = tier_stats[band]
        se = sig / math.sqrt(max(span, 0.25))
        wgt = tier_tau * tier_tau / (tier_tau * tier_tau + se * se)
        drift = wgt * raw_slope + (1.0 - wgt) * tier_mean
        drift = min(DRIFT_CLAMP_U[1], max(DRIFT_CLAMP_U[0], drift))
        # Forecast on the u scale, then convert back to rates.
        u_t = u_of_pct(last_rate)
        future_dil = 0.0
        if n_hat is not None and last_n > 0 and n_hat > last_n:
            diluted_u = math.log(1.0 - (last_rate / 100.0) * (last_n / n_hat))
            future_dil = diluted_u - u_t
        u_hat = u_t + horizon_days * drift + future_dil
        spread = sig * math.sqrt(max(horizon_days, 0.0))
        # Higher rate = lower u, so the 95% rate quantile is u_hat - z*spread.
        qs = [pct_of_u(u_hat - z * spread) for z in Z]
        qs = enforce_monotone(qs)
        # Bound the logit spread: a fixed u-spread maps to an extreme logit
        # spread at low rates. Cap each quantile's logit deviation from the
        # median (the baseline's own spread is about +-1.75 logit at this
        # horizon, so 2.5 leaves room for genuinely wider uncertainty).
        lm = logit_pct(qs[2])
        capped = []
        for q in qs:
            lq = logit_pct(q)
            lq = min(lm + 2.5, max(lm - 2.5, lq))
            capped.append(100.0 / (1.0 + math.exp(-lq)))
        qs = enforce_monotone(capped)
        return qs
    except Exception:
        return baseline_quantiles(last_rate, horizon_days)


def forecast_case(case, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    try:
        stats = tier_statistics(case)
    except Exception:
        stats = [(0.0, TAU_FLOOR)] * 4
    try:
        n_hat = players_growth_rate(case, horizon_days)
    except Exception:
        n_hat = None
    rows = []
    for j, trophy in enumerate(case["trophies"]):
        try:
            qs = forecast_trophy(case, j, horizon_days, stats, n_hat)
        except Exception:
            last = None
            series = trophy_rows(case, j)
            if series:
                last = series[-1][2]
            qs = baseline_quantiles(last, horizon_days)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case without output: baseline for every trophy.
            horizon_days = case.get("horizon_minutes", 0) / 1440.0
            fb = []
            for j, trophy in enumerate(case.get("trophies", [])):
                last = None
                try:
                    series = trophy_rows(case, j)
                    if series:
                        last = series[-1][2]
                except Exception:
                    last = None
                fb.append({"id": trophy["id"],
                           "q": baseline_quantiles(last, horizon_days)})
            forecast = {"trophies": fb}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

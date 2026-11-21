[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muhamadlangg71-prog/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muhamadlangg71-prog/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Entry for [make.fun's GTA VI trophy rarity forecast bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250): forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per trophy.

Referral: https://make.fun/r/sz08m7rk

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads every `<id>.in` in the input directory and writes `<id>.out`:

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

The model works with u = log(1 - p), the log share of players who have
NOT earned the trophy. Under a constant per-player hazard, u falls
linearly, so a linear extrapolation of u saturates gracefully in rate
space — a logit-linear drift would overshoot. It beats the house baseline
(persistence + 0.5/sqrt-day random-walk spread) by:

1. **Frozen-read collapse** — reads whose full state is identical to the
   previous read carry no information and are dropped from the fit.
2. **Dilution correction** — each interval's u move is split into the
   mechanical dilution from `players_tracked` growth and real progression;
   drift is estimated from the progression-adjusted series only.
3. **Shrunk drift** — per-trophy drift from a recency-weighted
   least-squares slope of the progression-adjusted u, shrunk toward the
   mean drift of its tier (data-driven bands of the last rate) with an
   empirical-Bayes weight, so flat or short history falls back toward the
   tier mean instead of trusting noise.
4. **Damped player growth** — future `players_tracked` is forecast with
   damped log-growth and the future dilution it implies is applied.
5. **Per-trophy spread** — from the observed variance of
   progression-adjusted u increments (floored, so flat history stays
   honestly uncertain).

Any trophy or case that fails inside the model falls back to the house
baseline, so the output is always valid.

Synthetic backtest (hazard-based simulation with rising rates and growing
player counts, scored by pinball loss on the logit scale): the model beats
the baseline by ~88-90% across scenarios (0.08-0.16 vs 0.62-1.35).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.

"""
===============================================================================
HMM Regime Detector — Universal Demo Script
===============================================================================
This script demonstrates how to use the HMMRegimeRadar module with ANY asset
by generating synthetic OHLCV data (random walk) and feeding it to the model
to obtain market regime state outputs.

Usage:
    conda activate hmm_detector_env
    python demo_universal.py

Expected output:
    - Feature matrix dimension and time range
    - Predicted posterior probabilities for PANIC / OSCILLATE / TREND states
    - Final regime action series (0=PANIC_OFF, 1=OBSERVE, 2=TREND_ON)
    - State distribution summary table

Author: Open-source contributor
License: MIT
===============================================================================
"""

import logging
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

# ---- Import the HMM Regime Radar module (must be in same directory) ----
# The module internally imports from data_pipeline_v2, but we NEVER call
# run_regime_radar() — instead we directly inject synthetic data via the
# low-level feature computation pipeline.
from hmm_regime_radar import HMMRegimeRadar


# ==============================================================================
# 1. Synthetic OHLCV Data Generator (Random Walk)
# ==============================================================================
def generate_synthetic_ohlcv(
    symbols: list,
    n_days: int = 750,
    seed: int = 42,
    freq: str = "B",
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for a list of symbols using a vectorized
    geometric Brownian motion (random walk) process.

    Parameters
    ----------
    symbols : list of str
        List of ticker symbols to generate data for.
    n_days : int
        Number of trading days to generate (default 750 ≈ 3 years).
    seed : int
        Random seed for reproducibility.
    freq : str
        Pandas frequency string (default 'B' = business daily).

    Returns
    -------
    Dict[str, pd.DataFrame]
        {symbol: DataFrame} with columns ['open','high','low','close','volume']
        and a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)

    # Create a DatetimeIndex ending at 'today'
    end_date = pd.Timestamp.today().normalize()
    date_index = pd.bdate_range(end=end_date, periods=n_days, freq=freq)
    n = len(date_index)

    panel: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        # ---- Daily log-returns: N(0.0004, 0.018^2) — roughly stock-like ----
        log_returns = rng.normal(loc=0.0004, scale=0.018, size=n)

        # ---- Cumulative price from initial 100 ----
        price = 100.0 * np.exp(np.cumsum(log_returns))

        # ---- Derive OHLC from daily close ----
        close = price
        open_ = np.zeros(n)
        high = np.zeros(n)
        low = np.zeros(n)

        # Vectorized OHLC simulation
        open_[0] = 100.0
        for i in range(1, n):
            open_[i] = close[i - 1]

        # High: close + random up-wick, Low: close - random down-wick
        daily_vol = np.abs(close) * 0.012
        high = np.maximum(close, open_) + rng.uniform(0, 1, size=n) * daily_vol
        low = np.minimum(close, open_) - rng.uniform(0, 1, size=n) * daily_vol

        # Ensure high >= open/close >= low
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # ---- Volume: lognormal with slight trend ----
        volume = rng.lognormal(mean=14.0, sigma=0.6, size=n).astype(np.int64)

        # ---- Assemble DataFrame ----
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=date_index,
        )

        panel[sym] = df

    return panel


# ==============================================================================
# 2. Main Demo
# ==============================================================================
def main():
    print("=" * 72)
    print("  HMM Regime Detector — Universal Demo")
    print("  Synthetic OHLCV Data → HMM Training → Regime States")
    print("=" * 72)

    # ---- Step 1: Configure logging (suppress verbose output) ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ---- Step 2: Generate synthetic data for 3 synthetic assets ----
    print("\n[Step 1] Generating synthetic OHLCV data (750 days, 3 assets)...")
    t0 = time.time()

    synthetic_symbols = ["SYNTH_A", "SYNTH_B", "SYNTH_C"]
    panel_dict = generate_synthetic_ohlcv(
        symbols=synthetic_symbols,
        n_days=750,
        seed=42,
    )

    elapsed_data = time.time() - t0
    print(
        f"         Generated {len(panel_dict)} assets × "
        f"{len(next(iter(panel_dict.values())))} days "
        f"({elapsed_data:.2f}s)"
    )
    print(f"         Date range: {list(panel_dict.values())[0].index[0].date()}"
          f" → {list(panel_dict.values())[0].index[-1].date()}")

    # ---- Step 3: Instantiate the HMM regime radar ----
    print("\n[Step 2] Initializing HMMRegimeRadar (window=504, step=20)...")
    radar = HMMRegimeRadar(
        etf_pool=synthetic_symbols,
        window=504,
        step=20,
    )

    # ---- Step 4: Compute cross-sectional features (raw, no normalization) ----
    print("\n[Step 3] Computing 4-dim cross-sectional features (F1~F4)...")
    t1 = time.time()
    feature_df = radar._compute_features(panel_dict)
    elapsed_feat = time.time() - t1
    print(f"         Feature matrix shape: {feature_df.shape}")
    print(f"         Columns: {list(feature_df.columns)}")
    print(f"         Time range: {feature_df.index[0].date()} "
          f"→ {feature_df.index[-1].date()}")
    print(f"         ({elapsed_feat:.2f}s)")

    # Show a peek of the feature values
    print("\n         Feature peek (last 5 rows):")
    print(feature_df.tail(5).to_string())

    # ---- Step 5: Rolling HMM training → forward algorithm → probabilities ----
    print("\n[Step 4] Running rolling-window HMM training + forward algorithm...")
    t2 = time.time()

    # The internal _rolling_hmm_train handles:
    #   - Windowed train/predict split (504 day window, 20 day step)
    #   - Per-window Winsorize + StandardScaler (no look-ahead)
    #   - GaussianHMM fit with state re-alignment (by F1 mean sorting)
    #   - Pure forward pass (no backward, no future data leakage)
    proba_df = radar._rolling_hmm_train(feature_df)

    elapsed_hmm = time.time() - t2
    print(f"         Posterior probability matrix shape: {proba_df.shape}")
    print(f"         Columns: {list(proba_df.columns)}")
    print(f"         Time range: {proba_df.index[0].date()} "
          f"→ {proba_df.index[-1].date()}")
    print(f"         ({elapsed_hmm:.2f}s)")

    # Show probability tail
    print("\n         Posterior probabilities (last 10 rows):")
    proba_tail = proba_df.tail(10).copy()
    proba_tail.index = proba_tail.index.strftime("%Y-%m-%d")
    print(proba_tail.to_string(float_format="%.4f"))

    # ---- Step 6: Hysteresis comparator (Schmitt trigger) ----
    print("\n[Step 5] Applying hysteresis comparator (Schmitt trigger)...")
    t3 = time.time()
    regime_series = radar._hysteresis_comparator(proba_df)
    elapsed_schmitt = time.time() - t3
    print(f"         Regime series length: {len(regime_series)}")
    print(f"         ({elapsed_schmitt:.4f}s)")

    # ---- Step 7: Results summary ----
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    # Regime tail
    print("\n  Regime actions (last 20 days):")
    print("  Encoding:  0 = PANIC_OFF | 1 = OBSERVE | 2 = TREND_ON")
    regime_tail = regime_series.tail(20).to_frame("regime_action")
    regime_tail.index = regime_tail.index.strftime("%Y-%m-%d")
    print(f"\n{regime_tail.to_string()}")

    # State distribution
    n_total = len(regime_series)
    cnt_0 = (regime_series == 0).sum()
    cnt_1 = (regime_series == 1).sum()
    cnt_2 = (regime_series == 2).sum()

    print("\n  State distribution:")
    print(f"    PANIC_OFF (0): {cnt_0:5d}  ({cnt_0 / n_total * 100:5.1f}%)")
    print(f"    OBSERVE   (1): {cnt_1:5d}  ({cnt_1 / n_total * 100:5.1f}%)")
    print(f"    TREND_ON  (2): {cnt_2:5d}  ({cnt_2 / n_total * 100:5.1f}%)")
    print(f"    ─────────────────────────────────")
    print(f"    TOTAL        : {n_total:5d}  (100.0%)")

    # Validate output contract
    print("\n  Data contract validation:")
    try:
        HMMRegimeRadar._validate_output(regime_series)
        print("    ✅ All checks passed: pd.Series, DatetimeIndex,"
              " regime_action name, values in {0,1,2}, no NaN")
    except AssertionError as e:
        print(f"    ❌ Validation failed: {e}")

    # Timing summary
    total_time = time.time() - t0
    print(f"\n  Total wall time: {total_time:.2f}s")
    print("=" * 72)
    print("Demo completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()

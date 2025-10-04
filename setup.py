"""
pricing_model.py
----------------
Robust prototype pricing model for natural gas storage contracts.

Key functions:
- load_price_series(csv_path, forecast_months=12)
- get_price_estimate(combined_series, date_input)
- price_multi_leg_storage(combined_series, injection_schedule, withdrawal_schedule, ...)
- price_contract_compat(in_dates, in_prices, out_dates, out_prices, rate, storage_cost_rate, total_vol, injection_withdrawal_cost_rate, ...)
- simulate_paths(...)  # optional Monte-Carlo simulator (simple OU around seasonal mean)

Author: ChatGPT (prototype)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# -------------------------
# Data loading & forecasting
# -------------------------
def load_price_series(csv_path: str = "data/Nat_Gas.csv", forecast_months: int = 12
                     ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load monthly price CSV and extend with Holt-Winters seasonal forecast.

    CSV must contain columns: ['Dates', 'Prices'] where Dates are month-end (or parseable).
    Returns (combined_series, historic_df) where combined_series index is monthly timestamps.
    """
    df = pd.read_csv(csv_path)
    if "Dates" not in df.columns or "Prices" not in df.columns:
        raise ValueError("CSV must contain 'Dates' and 'Prices' columns.")
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").set_index("Dates")
    # Ensure month-end aligned timestamps (period -> month end)
    df.index = df.index.to_period("M").to_timestamp("M")
    df["Prices"] = pd.to_numeric(df["Prices"], errors="coerce")
    df = df.dropna(subset=["Prices"])
    if df.empty:
        raise ValueError("No valid price data found in CSV.")

    # Fit Holt-Winters with additive trend & seasonality (period=12 months)
    model = ExponentialSmoothing(df["Prices"], trend="add", seasonal="add", seasonal_periods=12,
                                 initialization_method="estimated")
    fit = model.fit(optimized=True)
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=forecast_months, freq="M")
    forecast_series = pd.Series(fit.forecast(forecast_months), index=forecast_index)

    combined = pd.concat([df["Prices"], forecast_series]).sort_index()
    return combined, df


def _index_ordinals(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of ordinal indices and values for interpolation."""
    idx_ord = np.array([ts.toordinal() for ts in series.index.to_pydatetime()], dtype=float)
    vals = series.values.astype(float)
    return idx_ord, vals


# -------------------------
# Price lookup / estimator
# -------------------------
def get_price_estimate(combined_series: pd.Series, date_input: Any) -> float:
    """
    Estimate price for an arbitrary date using linear interpolation between month-end points.
    If outside the known range, linearly extrapolates using the closest two points.
    date_input: str, datetime.date, or pandas Timestamp.
    """
    # Normalize input date to date object
    if isinstance(date_input, str):
        dt = pd.to_datetime(date_input).date()
    elif isinstance(date_input, pd.Timestamp):
        dt = date_input.to_pydatetime().date()
    elif isinstance(date_input, datetime):
        dt = date_input.date()
    elif isinstance(date_input, date):
        dt = date_input
    else:
        raise TypeError("date_input must be str/date/datetime/pd.Timestamp")

    idx_ord, vals = _index_ordinals(combined_series)
    target_ord = float(dt.toordinal())

    if target_ord < idx_ord[0]:
        # extrapolate backward using first two points
        x0, x1 = idx_ord[0], idx_ord[1]
        y0, y1 = vals[0], vals[1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (target_ord - x0))
    if target_ord > idx_ord[-1]:
        # extrapolate forward using last two points
        x0, x1 = idx_ord[-2], idx_ord[-1]
        y0, y1 = vals[-2], vals[-1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (target_ord - x0))

    return float(np.interp(target_ord, idx_ord, vals))


# -------------------------
# Helpers
# -------------------------
def _to_date_obj(d: Any) -> date:
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    return pd.to_datetime(d).date()


def _months_between(d1: date, d2: date) -> int:
    if d2 < d1:
        return 0
    return max(0, (d2.year - d1.year) * 12 + (d2.month - d1.month))


# -------------------------
# Pricing engine (event-based)
# -------------------------
def price_multi_leg_storage(
    combined_series: pd.Series,
    injection_schedule: List[Dict],
    withdrawal_schedule: List[Dict],
    injection_rate_mmbtu_per_month: float,
    withdrawal_rate_mmbtu_per_month: float,
    max_storage_mmbtu: float,
    storage_fee_fixed_per_month: float = 0.0,
    storage_fee_per_mmbtu_per_month: float = 0.0,
    include_terminal_inventory_valuation: bool = True,
    pro_rata_storage_by_days: bool = False,
) -> Dict[str, Any]:
    """
    Robust event-driven valuation.

    injection_schedule / withdrawal_schedule: lists of dicts
      each dict: {'date': 'YYYY-MM-DD' or date, 'volume': float | None}
      If 'volume' is None, the model will use the rate cap for that event.

    Returns dict:
      {
        'initial_inventory': 0.0,
        'final_inventory': ...,
        'cashflows': [ {date,type,volume,unit_price,cashflow,detail}, ... ],
        'net_value': float
      }
    """
    # Build events preserving multiplicity & order by date
    events = []
    for ev in injection_schedule:
        events.append({"date": _to_date_obj(ev["date"]), "type": "inject", "requested": None if ev.get("volume") is None else float(ev["volume"])})
    for ev in withdrawal_schedule:
        events.append({"date": _to_date_obj(ev["date"]), "type": "withdraw", "requested": None if ev.get("volume") is None else float(ev["volume"])})
    if not events:
        raise ValueError("At least one injection or withdrawal event required.")

    events.sort(key=lambda x: x["date"])  # stable sort keeps same-date order

    inventory = 0.0
    cashflows: List[Dict[str, Any]] = []
    last_event_date = events[0]["date"]

    for ev in events:
        ev_date = ev["date"]

        # Accrue storage fees between last_event_date and ev_date
        if ev_date > last_event_date and (storage_fee_fixed_per_month or storage_fee_per_mmbtu_per_month):
            if pro_rata_storage_by_days:
                days = (ev_date - last_event_date).days
                months_equiv = days / 30.0
                fixed = storage_fee_fixed_per_month * months_equiv
                variable = storage_fee_per_mmbtu_per_month * inventory * months_equiv
                months_recorded = months_equiv
            else:
                months = _months_between(last_event_date, ev_date)
                fixed = storage_fee_fixed_per_month * months
                variable = storage_fee_per_mmbtu_per_month * inventory * months
                months_recorded = months
            total_storage_fee = fixed + variable
            if abs(total_storage_fee) > 0:
                cashflows.append({
                    "date": last_event_date + timedelta(days=1),
                    "type": "storage_fee",
                    "months_equiv": months_recorded,
                    "inventory": inventory,
                    "cashflow": -total_storage_fee,
                    "detail": {"fixed": fixed, "variable": variable}
                })

        # Determine price for this event using combined_series (interpolate/extrapolate)
        price = get_price_estimate(combined_series, ev_date)

        if ev["type"] == "inject":
            requested = ev["requested"] if ev["requested"] is not None else injection_rate_mmbtu_per_month
            # capacity and rate constraints
            can_inject = min(injection_rate_mmbtu_per_month, max_storage_mmbtu - inventory)
            actual = min(requested, can_inject)
            if actual <= 0:
                cashflows.append({
                    "date": ev_date, "type": "inject_failed", "requested": requested, "actual": 0.0,
                    "inventory": inventory, "cashflow": 0.0, "note": "no capacity or rate 0"
                })
            else:
                buy_cost = actual * price
                injection_cost = actual * (0.0)  # injection/withdrawal per-unit cost should be a parameter if needed
                total_out = -(buy_cost + injection_cost)
                inventory += actual
                cashflows.append({
                    "date": ev_date,
                    "type": "inject",
                    "volume": actual,
                    "unit_price": price,
                    "cashflow": total_out,
                    "detail": {"buy_cost": buy_cost, "injection_cost": injection_cost, "inventory_after": inventory}
                })
        else:  # withdraw
            requested = ev["requested"] if ev["requested"] is not None else withdrawal_rate_mmbtu_per_month
            can_withdraw = min(withdrawal_rate_mmbtu_per_month, inventory)
            actual = min(requested, can_withdraw)
            if actual <= 0:
                cashflows.append({
                    "date": ev_date, "type": "withdraw_failed", "requested": requested, "actual": 0.0,
                    "inventory": inventory, "cashflow": 0.0, "note": "insufficient inventory"
                })
            else:
                revenue = actual * price
                withdrawal_cost = actual * (0.0)  # similar: leave 0 here; use price_contract_compat for per-unit transfer cost
                net_in = revenue - withdrawal_cost
                inventory -= actual
                cashflows.append({
                    "date": ev_date,
                    "type": "withdraw",
                    "volume": actual,
                    "unit_price": price,
                    "cashflow": net_in,
                    "detail": {"revenue": revenue, "withdrawal_cost": withdrawal_cost, "inventory_after": inventory}
                })

        last_event_date = ev_date

    # Terminal inventory valuation
    if include_terminal_inventory_valuation and inventory > 0:
        last_price = get_price_estimate(combined_series, combined_series.index[-1])
        terminal_value = inventory * last_price
        cashflows.append({
            "date": combined_series.index[-1],
            "type": "terminal_valuation",
            "volume": inventory,
            "unit_price": last_price,
            "cashflow": terminal_value
        })

    net_value = sum(c["cashflow"] for c in cashflows)
    return {
        "initial_inventory": 0.0,
        "final_inventory": inventory,
        "cashflows": cashflows,
        "net_value": net_value
    }


# -------------------------
# Compatibility wrapper (user-provided style)
# -------------------------
def price_contract_compat(
    in_dates: List[Any],
    in_prices: List[float],
    out_dates: List[Any],
    out_prices: List[float],
    rate: float,
    storage_cost_rate: float,
    total_vol: float,
    injection_withdrawal_cost_rate: float,
    storage_fee_per_mmbtu_per_month: float = 0.0,
    pro_rata_storage_by_days: bool = False,
    include_terminal_inventory_valuation: bool = True
) -> Dict[str, Any]:
    """
    Compatibility wrapper that accepts the same inputs as your original function,
    but uses the robust engine to compute all cashflows and returns an object with 'net_value'.

    - in_dates/out_dates: list of date strings or date objects
    - in_prices/out_prices: lists of per-event prices (same length as corresponding dates)
    - rate: default per-event volume if 'volume' unspecified (we treat each event volume = rate)
    - storage_cost_rate: fixed $ per month (applied as before)
    - injection_withdrawal_cost_rate: $ per unit transferred
    """
    # Build schedules in the event format expected by price_multi_leg_storage
    injection_schedule = []
    withdrawal_schedule = []

    # Normalize inputs
    if not (len(in_dates) == len(in_prices) and len(out_dates) == len(out_prices)):
        # It's allowed to have differing counts between injections and withdrawals, but each list must pair price->date
        pass

    for d, p in zip(in_dates, in_prices):
        injection_schedule.append({"date": d, "volume": rate, "unit_price": float(p)})

    for d, p in zip(out_dates, out_prices):
        withdrawal_schedule.append({"date": d, "volume": rate, "unit_price": float(p)})

    # We will use a combined_series built from the provided event prices to ensure price lookup uses those values.
    # Create a synthetic monthly series that includes all event dates and their prices (month-end aligned).
    # Simpler approach: build a pandas Series keyed by event dates (month-end normalized) using provided prices,
    # then fill remaining months by linear interpolation. This is a pragmatic hybrid to ensure event prices are honored.
    all_events = []
    for d, p in zip(in_dates, in_prices):
        all_events.append((_to_date_obj(d), float(p)))
    for d, p in zip(out_dates, out_prices):
        all_events.append((_to_date_obj(d), float(p)))
    if not all_events:
        raise ValueError("No event prices provided.")

    # Create monthly index covering min->max event months
    min_date = min(d for d, _ in all_events)
    max_date = max(d for d, _ in all_events)
    monthly_index = pd.date_range(start=pd.to_datetime(min_date).to_period("M").to_timestamp("M"),
                                  end=pd.to_datetime(max_date).to_period("M").to_timestamp("M"),
                                  freq="M")
    # Create series with NaNs and then set event months to provided prices (last event price for month if multiple)
    s = pd.Series(index=monthly_index, dtype=float)
    for d, p in all_events:
        m = pd.to_datetime(d).to_period("M").to_timestamp("M")
        s.loc[m] = p
    # Interpolate remaining months linearly
    s = s.interpolate(method="linear").ffill().bfill()

    # Extend with a 12-month HW forecast on top of this synthetic series
    # Fit Holt-Winters to s (needs at least 2 years ideally, but we still attempt)
    try:
        hw = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
        hw_fit = hw.fit(optimized=True)
        forecast_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
        forecast_s = pd.Series(hw_fit.forecast(12), index=forecast_idx)
        combined_series = pd.concat([s, forecast_s]).sort_index()
    except Exception:
        # If HW fails (too short series), fallback to extending last observed price flat
        forecast_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
        forecast_s = pd.Series([s.iloc[-1]] * len(forecast_idx), index=forecast_idx)
        combined_series = pd.concat([s, forecast_s]).sort_index()

    # Convert schedules to the injection/withdrawal dicts used by the engine, using actual requested volumes = rate
    inj_sched = [{"date": ev["date"], "volume": ev["volume"]} for ev in injection_schedule]
    wdr_sched = [{"date": ev["date"], "volume": ev["volume"]} for ev in withdrawal_schedule]

    # Use the robust engine. The engine does not currently apply per-transfer injection_withdrawal_cost_rate,
    # so we'll post-process inject/withdraw events to include those costs in cashflow detail.
    result = price_multi_leg_storage(
        combined_series=combined_series,
        injection_schedule=inj_sched,
        withdrawal_schedule=wdr_sched,
        injection_rate_mmbtu_per_month=rate,
        withdrawal_rate_mmbtu_per_month=rate,
        max_storage_mmbtu=total_vol,
        storage_fee_fixed_per_month=storage_cost_rate,
        storage_fee_per_mmbtu_per_month=storage_fee_per_mmbtu_per_month if 'storage_fee_per_mmbtu_per_month' in globals() else 0.0,
        include_terminal_inventory_valuation=include_terminal_inventory_valuation,
        pro_rata_storage_by_days=pro_rata_storage_by_days
    )

    # Inject/withdraw transfer costs (per-unit) — adjust cashflows to include those explicit costs
    adjusted_cashflows = []
    for cf in result["cashflows"]:
        cf2 = cf.copy()
        if cf2.get("type") == "inject":
            transfer_cost = cf2["volume"] * injection_withdrawal_cost_rate
            cf2["cashflow"] -= transfer_cost
            cf2.setdefault("detail", {})["injection_transfer_cost"] = transfer_cost
        if cf2.get("type") == "withdraw":
            transfer_cost = cf2["volume"] * injection_withdrawal_cost_rate
            cf2["cashflow"] -= transfer_cost
            cf2.setdefault("detail", {})["withdrawal_transfer_cost"] = transfer_cost
        adjusted_cashflows.append(cf2)

    net_value = sum(c["cashflow"] for c in adjusted_cashflows)
    return {
        "final_inventory": result["final_inventory"],
        "cashflows": adjusted_cashflows,
        "net_value": net_value
    }


# -------------------------
# Optional: Monte Carlo simulator (simple OU around seasonal mean)
# -------------------------
def simulate_paths(
    combined_series: pd.Series,
    n_paths: int = 500,
    horizon_months: int = 12,
    dt_months: float = 1.0,
    theta: float = 1.0,
    sigma: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate price paths using simple discrete Ornstein-Uhlenbeck around seasonal mean.
    - combined_series: historical+forecast monthly series (used to extract seasonal mean by month)
    Returns array of shape (n_paths, horizon_months) with simulated monthly prices.
    """
    rng = np.random.default_rng(seed)
    # Seasonal mean by month (1..12)
    by_month = pd.Series(combined_series.values, index=combined_series.index).groupby(combined_series.index.month).mean()
    last_date = combined_series.index[-1]
    months = [(last_date + pd.offsets.MonthEnd(i)).month for i in range(1, horizon_months + 1)]
    mu = np.array([by_month[m] for m in months], dtype=float)

    # Initialize at last observed price
    S0 = float(combined_series.iloc[-1])
    paths = np.zeros((n_paths, horizon_months), dtype=float)

    for i in range(n_paths):
        s = S0
        for t in range(horizon_months):
            # OU discrete step: s_{t+1} = s_t + theta*(mu_t - s_t)*dt + sigma*sqrt(dt)*Z
            z = rng.standard_normal()
            s = s + theta * (mu[t] - s) * dt_months + sigma * math.sqrt(dt_months) * z
            paths[i, t] = max(0.0, s)
    return paths


# -------------------------
# Example usage when script run directly
# -------------------------
if __name__ == "__main__":
    # Quick demo using data/Nat_Gas.csv in project root
    try:
        combined, hist = load_price_series("data/Nat_Gas(1).csv", forecast_months=12)
    except Exception as e:
        raise SystemExit(f"Failed to load data: {e}")

    print("Loaded price series (last rows):")
    print(combined.tail())

    # Example: user-style contract (compat wrapper)
    from datetime import date
    in_dates = [date(2024, 10, 31), date(2024, 11, 30)]
    in_prices = [get_price_estimate(combined, d) for d in in_dates]
    out_dates = [date(2025, 2, 28)]
    out_prices = [get_price_estimate(combined, out_dates[0])]

    res = price_contract_compat(
        in_dates=in_dates,
        in_prices=in_prices,
        out_dates=out_dates,
        out_prices=out_prices,
        rate=500_000,
        storage_cost_rate=100_000,
        total_vol=1_200_000,
        injection_withdrawal_cost_rate=0.01
    )

    print("\nExample contract result:")
    print("Net value:", res["net_value"])
    for c in res["cashflows"]:
        print(c)


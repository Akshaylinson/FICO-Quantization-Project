
# prototype_pricing_model.py
# Prototype multi-leg natural gas storage pricing model
# - Loads monthly price CSV (Nat_Gas.csv)
# - Builds a 12-month seasonal forecast (Holt-Winters)
# - Exposes get_price_estimate(date) and price_multi_leg_storage(...) functions
# - Includes example scenarios and prints results when run as __main__

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import List, Dict
import json

CSV_PATH = '/mnt/data/Nat_Gas.csv'
FORECAST_MONTHS = 12

def load_price_series(csv_path=CSV_PATH, forecast_months=FORECAST_MONTHS):
    df = pd.read_csv(csv_path)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df = df.sort_values('Dates').set_index('Dates')
    df.index = df.index.to_period('M').to_timestamp('M')
    df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce')
    # Fit Holt-Winters seasonal model and forecast
    hw = ExponentialSmoothing(df['Prices'], trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated')
    fit = hw.fit(optimized=True)
    forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=forecast_months, freq='M')
    forecast_series = pd.Series(fit.forecast(forecast_months), index=forecast_index)
    combined = pd.concat([df['Prices'], forecast_series])
    return combined, df

def get_price_estimate(combined_series, date_input):
    dt = pd.to_datetime(date_input)
    idx_ord = np.array([ts.toordinal() for ts in combined_series.index.to_pydatetime()])
    vals = combined_series.values.astype(float)
    target = dt.toordinal()
    if target < idx_ord[0] or target > idx_ord[-1]:
        # linear extrapolation
        if target < idx_ord[0]:
            x0, x1 = idx_ord[0], idx_ord[1]
            y0, y1 = vals[0], vals[1]
        else:
            x0, x1 = idx_ord[-2], idx_ord[-1]
            y0, y1 = vals[-2], vals[-1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (target - x0))
    return float(np.interp(target, idx_ord, vals))

def price_multi_leg_storage(
    combined_series,
    injection_schedule: List[Dict],
    withdrawal_schedule: List[Dict],
    injection_rate_mmbtu_per_month: float,
    withdrawal_rate_mmbtu_per_month: float,
    max_storage_mmbtu: float,
    storage_fee_fixed_per_month: float = 0.0,
    storage_fee_per_mmbtu_per_month: float = 0.0,
    include_terminal_inventory_valuation: bool = True
):
    """Prototype multi-leg storage valuation.
    - injection_schedule & withdrawal_schedule: lists of {'date': 'YYYY-MM-DD', 'volume': float|None (MMBtu)}
      If volume is None, the model injects/withdraws up to the rate cap (or available capacity/inventory).
    - Rates and fees applied on a per-month basis. No discounting (interest=0).
    Returns a dict with cashflows and net_value.
    """
    # build unified event list
    events = []
    for it in injection_schedule:
        events.append({'date': pd.to_datetime(it['date']), 'type': 'inject', 'requested': None if it.get('volume') is None else float(it['volume'])})
    for it in withdrawal_schedule:
        events.append({'date': pd.to_datetime(it['date']), 'type': 'withdraw', 'requested': None if it.get('volume') is None else float(it['volume'])})
    if not events:
        raise ValueError("At least one injection or withdrawal event required.")
    events.sort(key=lambda x: x['date'])
    # state
    inventory = 0.0
    cashflows = []
    last_event_date = events[0]['date']
    def months_between(d1, d2):
        return max(0, (d2.year - d1.year)*12 + (d2.month - d1.month))
    for ev in events:
        ev_date = ev['date']
        # accrue storage fees between last_event_date and ev_date
        months = months_between(last_event_date, ev_date)
        if months > 0 and (storage_fee_fixed_per_month or storage_fee_per_mmbtu_per_month):
            fixed = storage_fee_fixed_per_month * months
            variable = storage_fee_per_mmbtu_per_month * inventory * months
            total_storage = fixed + variable
            cashflows.append({'date': last_event_date + pd.Timedelta(days=1), 'type': 'storage_fee', 'months': months, 'inventory': inventory, 'cashflow': -total_storage})
        price = get_price_estimate(combined_series, ev_date)
        if ev['type'] == 'inject':
            requested = ev['requested'] if ev['requested'] is not None else injection_rate_mmbtu_per_month
            can_inject = min(injection_rate_mmbtu_per_month, max_storage_mmbtu - inventory)
            actual = min(requested, can_inject)
            if actual > 0:
                cash = - actual * price
                inventory += actual
                cashflows.append({'date': ev_date, 'type': 'inject', 'volume': actual, 'unit_price': price, 'cashflow': cash})
            else:
                cashflows.append({'date': ev_date, 'type': 'inject_failed', 'volume': 0.0, 'unit_price': price, 'cashflow': 0.0})
        else:  # withdraw
            requested = ev['requested'] if ev['requested'] is not None else withdrawal_rate_mmbtu_per_month
            can_withdraw = min(withdrawal_rate_mmbtu_per_month, inventory)
            actual = min(requested, can_withdraw)
            if actual > 0:
                cash = actual * price
                inventory -= actual
                cashflows.append({'date': ev_date, 'type': 'withdraw', 'volume': actual, 'unit_price': price, 'cashflow': cash})
            else:
                cashflows.append({'date': ev_date, 'type': 'withdraw_failed', 'volume': 0.0, 'unit_price': price, 'cashflow': 0.0})
        last_event_date = ev_date
    # terminal valuation if requested
    terminal_value = 0.0
    if include_terminal_inventory_valuation and inventory > 0:
        last_price = get_price_estimate(combined_series, combined_series.index[-1])
        terminal_value = inventory * last_price
        cashflows.append({'date': combined_series.index[-1], 'type': 'terminal_valuation', 'volume': inventory, 'unit_price': last_price, 'cashflow': terminal_value})
    net = sum([c['cashflow'] for c in cashflows])
    return {'initial_inventory': 0.0, 'final_inventory': inventory, 'cashflows': cashflows, 'net_value': net}

# Example runner for quick tests
def example_run():
    combined, df = load_price_series()
    # Scenario A: explicit injections across two months, withdraw later in winter
    inj = [
        {'date': (df.index[-1] + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d'), 'volume': 500_000},
        {'date': (df.index[-1] + pd.offsets.MonthEnd(2)).strftime('%Y-%m-%d'), 'volume': 500_000},
    ]
    wdr = [
        {'date': (df.index[-1] + pd.offsets.MonthEnd(5)).strftime('%Y-%m-%d'), 'volume': 1_000_000},
    ]
    out = price_multi_leg_storage(combined, inj, wdr,
                                  injection_rate_mmbtu_per_month=600_000,
                                  withdrawal_rate_mmbtu_per_month=1_000_000,
                                  max_storage_mmbtu=1_200_000,
                                  storage_fee_fixed_per_month=100_000,
                                  storage_fee_per_mmbtu_per_month=0.0,
                                  include_terminal_inventory_valuation=False)
    print('\nScenario A result:')
    print(json.dumps(out, indent=2, default=str))
    # Scenario B: rate-driven injection/withdrawal (None volumes -> use rate caps)
    inj2 = [
        {'date': (df.index[-1] + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d'), 'volume': None},
        {'date': (df.index[-1] + pd.offsets.MonthEnd(2)).strftime('%Y-%m-%d'), 'volume': None},
    ]
    wdr2 = [
        {'date': (df.index[-1] + pd.offsets.MonthEnd(4)).strftime('%Y-%m-%d'), 'volume': None},
        {'date': (df.index[-1] + pd.offsets.MonthEnd(5)).strftime('%Y-%m-%d'), 'volume': None},
    ]
    out2 = price_multi_leg_storage(combined, inj2, wdr2,
                                   injection_rate_mmbtu_per_month=400_000,
                                   withdrawal_rate_mmbtu_per_month=600_000,
                                   max_storage_mmbtu=800_000,
                                   storage_fee_fixed_per_month=50_000,
                                   storage_fee_per_mmbtu_per_month=0.02,
                                   include_terminal_inventory_valuation=True)
    print('\nScenario B result:')
    print(json.dumps(out2, indent=2, default=str))

if __name__ == '__main__':
    example_run()


"""HydroSteward - data-center cooling module (SIMULATED).

A small 1 MW site with hybrid cooling: evaporative cooling (uses water) plus dry coolers (use almost none).
HydroSteward forecasts the next hour, offloads as much heat as is SAFE to the dry coolers, and keeps
the evaporative stage running as a floor. Values marked ASSUMED are assumptions, not measurements.
Run:  python hydrosteward_cooling.py   ->  writes cooling_data.json
"""
import json
import numpy as np

rng = np.random.default_rng(11)

WUE_EVAP = 1.9       # litres of water per kWh of heat removed by evaporative cooling (EESI average for data centers)
CAP_EVAP = 1500.0    # kW, ASSUMED evaporative capacity, always available
DRY_RATED = 500.0    # kW, ASSUMED dry-cooler capacity at 15 C
T_REF, T_MAX = 15.0, 35.0   # ASSUMED: dry coolers lose capacity in a straight line, reaching zero at 35 C
FLOOR = 0.30         # safety floor: the evaporative stage always covers at least 30% of the load
T_BUF = 1.5          # ASSUMED safety buffer: plan as if it is 1.5 C hotter than forecast
DAYS = 7
HEAT_DAY, BURST_DAY = 4, 2   # day index of a heat wave (+6 C) and of an AI-workload burst (+200 kW for 3 h)


def dry_cap(temp, rated=DRY_RATED):
    return rated * np.clip((T_MAX - temp) / (T_MAX - T_REF), 0, 1)


def make_day(heat=0.0, burst=False):
    h = np.arange(24)
    amb = 24 + 6 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.0, 24) + rng.normal(0, 1.5) + heat  # ASSUMED climate
    load = 900 + 100 * np.sin(2 * np.pi * (h - 10) / 24) + rng.normal(0, 30, 24)                           # ASSUMED IT load, kW
    if burst:
        s = rng.integers(8, 18)
        load[s:s + 3] += 200
    return amb, np.clip(load, 600, 1250)


# ---------- Learn the normal pattern from 14 days of history ----------
hist = [make_day() for _ in range(14)]
prof_amb = np.mean([a for a, _ in hist], axis=0)
prof_load = np.mean([l for _, l in hist], axis=0)


def forecast(actual, profile, k):
    """Hourly profile plus a share k of the previous hour's surprise (learned baseline + correction).
    Tested on 2,000 simulated days: k = 0.6 cuts ambient error (1.54 to 1.10 C), but any k > 0 hurts the
    load forecast here, because the simulated load noise is independent from hour to hour (so k = 0)."""
    surprise = actual - profile
    return profile + k * np.r_[0.0, surprise[:-1]]


def run_day(amb, load, rated):
    fc_amb, fc_load = forecast(amb, prof_amb, 0.6), forecast(load, prof_load, 0.0)
    planned = np.minimum(dry_cap(fc_amb + T_BUF, rated), (1 - FLOOR) * fc_load)   # heat to hand to the dry coolers
    dry = np.minimum(np.minimum(dry_cap(amb, rated), planned), (1 - FLOOR) * load)  # what they deliver; floor held on ACTUAL load
    evap = load - dry                                                            # evaporative stage covers the rest
    return fc_amb, fc_load, dry, evap


def simulate(rated):
    out = []
    for d in range(DAYS):
        amb, load = make_day(heat=6.0 if d == HEAT_DAY else 0.0, burst=(d == BURST_DAY))
        out.append((amb, load) + run_day(amb, load, rated))
    return out


# ---------- Main scenario ----------
rng_state = rng.bit_generator.state
days_out, base_tot, hs_tot, short, min_share = [], 0.0, 0.0, 0, 1.0
mae_a, mae_l = [], []
for d, (amb, load, fa, fl, dry, evap) in enumerate(simulate(DRY_RATED)):
    base, hs = WUE_EVAP * load, WUE_EVAP * evap            # litres per hour (1 hour at kW = kWh)
    base_tot += base.sum()
    hs_tot += hs.sum()
    short += int((evap > CAP_EVAP).sum())
    min_share = min(min_share, float((evap / load).min()))
    mae_a.append(np.abs(fa - amb).mean())
    mae_l.append(np.abs(fl - load).mean())
    days_out.append({"day": d + 1, "note": "heat wave" if d == HEAT_DAY else ("AI workload burst" if d == BURST_DAY else "normal"),
                     "ambient_c": [round(float(x), 1) for x in amb], "load_kw": [round(float(x)) for x in load],
                     "dry_kw": [round(float(x)) for x in dry], "water_baseline_l": [round(float(x)) for x in base],
                     "water_hydrosteward_l": [round(float(x)) for x in hs]})

# ---------- Sensitivity: how much does it depend on the dry-cooler capacity we assumed? ----------
sens = []
for rated in (250.0, 500.0, 800.0):
    rng.bit_generator.state = rng_state
    runs = simulate(rated)
    b = sum(WUE_EVAP * l.sum() for _, l, *_ in runs)
    h = sum(WUE_EVAP * e.sum() for *_, e in runs)
    sens.append({"dry_rated_kw": rated, "saved_pct": round(100 * (1 - h / b), 1)})

summary = {"baseline_l_per_day": round(base_tot / DAYS), "hydrosteward_l_per_day": round(hs_tot / DAYS),
           "saved_l_per_day": round((base_tot - hs_tot) / DAYS), "saved_pct": round(100 * (1 - hs_tot / base_tot), 1),
           "shortfall_hours": short, "min_evaporative_share": round(min_share, 2),
           "forecast_mae_ambient_c": round(float(np.mean(mae_a)), 2), "forecast_mae_load_kw": round(float(np.mean(mae_l))),
           "effective_wue_l_per_kwh": round(hs_tot / sum(sum(d["load_kw"]) for d in days_out), 2)}
with open("cooling_data.json", "w") as f:
    json.dump({"meta": {"scenario": "SIMULATED 1 MW hybrid-cooled data center, 7 days", "wue_evap": WUE_EVAP,
                        "dry_rated_kw": DRY_RATED, "evap_floor": FLOOR},
               "days": days_out, "summary": summary, "sensitivity": sens}, f, indent=1)

print("summary:", summary)
for d, (b, h) in enumerate(zip([sum(x["water_baseline_l"]) for x in days_out], [sum(x["water_hydrosteward_l"]) for x in days_out])):
    print("day", d + 1, days_out[d]["note"], "| baseline", b, "L | HydroSteward", h, "L | saved", round(100 * (1 - h / b), 1), "%")
print("sensitivity:", sens)

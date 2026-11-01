"""HydroSteward - simulated 30-day scenario exported for the dashboard.

Everything here is SIMULATED. Parameters marked ASSUMED are assumptions, not measurements.
Run:  python hydrosteward_export.py   ->  writes hydrosteward_data.json
"""
import datetime
import json
import numpy as np

rng = np.random.default_rng(7)

# ---------- Parameters ----------
FLOW_LPM = 7.6          # 2.0 gpm bathroom faucet (EPA WaterSense, via Home Depot Eco Actions)
SIG = 0.05              # ASSUMED sensor noise, L/min
LEAK_LPM = 0.1          # ASSUMED leak size, L/min
DAYS = 30               # length of the simulated scenario
TAP_DAY = 9             # day index of the tap-left-running incident (starts 03:00)
LEAK_START = 19         # day index where the leak begins
START_DATE = datetime.date(2026, 10, 1)
w = np.array([1, 1, 1, 1, 2, 5, 9, 8, 5, 3, 3, 4, 5, 4, 3, 3, 4, 6, 8, 7, 5, 3, 2, 1])  # ASSUMED hourly rhythm
p = w / w.sum()


# ---------- Simulator ----------
def simulate_day(rng, n_uses=400):
    h = rng.choice(24, size=n_uses, p=p)
    start = h * 60 + rng.uniform(0, 60, size=n_uses)      # minutes since midnight
    dur = rng.uniform(10, 40, size=n_uses) / 60           # duration in minutes
    day = np.zeros(1440)
    for s, d in zip(start, dur):
        for m in range(int(s), min(int(s + d) + 1, 1440)):
            day[m] += FLOW_LPM * (min(s + d, m + 1) - max(s, m))
    return day


def noisy(day):
    return np.clip(day + rng.normal(0, SIG, 1440), 0, None)


# ---------- Detector: learned from 30 normal days ----------
history = np.array([simulate_day(rng) for _ in range(30)])
hourly_hist = history.reshape(30, 24, 60).sum(axis=2)
mean_h, std_h = hourly_hist.mean(axis=0), hourly_hist.std(axis=0)


def hourly_z(day):
    return (day.reshape(24, 60).sum(axis=1) - mean_h) / std_h


def min_flow(day, k=30):
    avg = np.convolve(day, np.ones(k) / k, mode="valid")  # 30-minute moving average
    return avg.min()                                      # quietest half-hour


quiet = np.array([min_flow(noisy(simulate_day(rng))) for _ in range(30)])
threshold = quiet.mean() + 3 * quiet.std()


def longest_run(x, thr):
    best = run = 0
    for v in x:
        run = run + 1 if v > thr else 0
        best = max(best, run)
    return best


def first_alarm(day, thr, n_min):
    run = 0
    for m, v in enumerate(day):
        run = run + 1 if v > thr else 0
        if run >= n_min:
            return m
    return None


longest = max(longest_run(d, 5) for d in history)
N = 2 * longest                                           # idle-flow alarm delay, minutes


def leak_alarm(day):
    return min_flow(day) > threshold


# ---------- Scenario ----------
records, alerts = [], []
leak_active, repair_day, prev_leak_alarm = False, None, False
tap_saved = 0.0

for i in range(DAYS):
    date = START_DATE + datetime.timedelta(days=i)
    day = simulate_day(rng)
    if i == LEAK_START:
        leak_active = True
    if repair_day is not None and i == repair_day:
        leak_active = False
    if leak_active:
        day = day + LEAK_LPM
    total_without = None
    if i == TAP_DAY:
        day[180:270] += FLOW_LPM                          # tap left running 03:00-04:30
        total_without = float(day.sum())
        m = first_alarm(day, 5, N)
        if m is not None:
            day[m + 1:270] -= FLOW_LPM                    # valve closes at the alarm
            saved = total_without - float(day.sum())
            tap_saved += saved
            alerts.append({"date": str(date), "time": "%02d:%02d" % (m // 60, m % 60),
                           "type": "Tap left running", "confirmed": True,
                           "detail": "Flow stayed high for %d min; valve closed; %.0f L saved" % (N, saved)})
    reading = noisy(day)
    la = bool(leak_alarm(reading))
    if la and prev_leak_alarm:
        alerts.append({"date": str(date), "time": "daily check", "type": "Possible leak", "confirmed": True,
                       "detail": "Night-time flow stayed above normal on two days in a row; repair requested"})
        if repair_day is None and leak_active:
            repair_day = i + 1                            # ASSUMED: repaired the next day
    elif la:
        alerts.append({"date": str(date), "time": "daily check", "type": "Leak watch", "confirmed": False,
                       "detail": "Night-time flow slightly raised; waiting for a second day to confirm"})
    prev_leak_alarm = la
    records.append({"date": str(date),
                    "litres": round(float(day.sum()), 1),
                    "litres_without_system": round(total_without if total_without is not None else float(day.sum()), 1),
                    "hourly": [round(float(x), 1) for x in day.reshape(24, 60).sum(axis=1)],
                    "quietest_30min_lpm": round(float(min_flow(reading)), 4),
                    "leak_active": bool(leak_active)})

leak_saved = 0.0
if repair_day is not None:
    leak_saved = (DAYS - repair_day) * LEAK_LPM * 1440    # litres the leak would have wasted until day 30

out = {
    "meta": {"scenario": "SIMULATED 30-day scenario", "flow_lpm": FLOW_LPM, "sensor_noise_lpm": SIG,
             "leak_lpm": LEAK_LPM, "leak_threshold_lpm": round(float(threshold), 4), "idle_alarm_minutes": int(N)},
    "days": records,
    "alerts": alerts,
    "summary": {"tap_saved_l": round(tap_saved, 1), "leak_saved_l": round(leak_saved, 1),
                "total_saved_l": round(tap_saved + leak_saved, 1),
                "total_used_l": round(sum(r["litres"] for r in records), 1)},
}
with open("hydrosteward_data.json", "w") as f:
    json.dump(out, f, indent=1)

print("threshold:", round(float(threshold), 4), "L/min | idle alarm after", N, "min")
print("alerts:", len(alerts), "| confirmed:", sum(a["confirmed"] for a in alerts))
for a in alerts:
    print(" ", a["date"], a["time"], a["type"], "-", "confirmed" if a["confirmed"] else "unconfirmed")
print("summary:", out["summary"])

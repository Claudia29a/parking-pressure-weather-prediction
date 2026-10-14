"""Step 5: predict parking for the next 7 days (run once a day).

Input:  data/processed/sessions_clean.csv    (recent usage level)
        models/poisson_B.joblib              (run src/train.py first)
        MET Norway forecast                  (downloaded live)
Output: data/forecasts/raw/<run>_<city>.json (every forecast is archived)
        data/forecasts/weather_forecast_daily.csv  (appended each run)
        data/predictions/predictions.csv     (appended each run)
        reports/figures/forecast_latest.png

What it does:
1. Downloads the MET Norway forecast for each city and archives it.
2. Turns it into the SAME daily weather as the training data: rain = >= 1 mm
   between 07-21h Dutch time, temp = mean temperature 07-21h.
3. Builds the 7 rows per city the model needs (weekday, usage level, weather).
4. Predicts sessions per day, saves the predictions and draws a chart.

The archives are what step 6 needs: later we compare these predictions (made
with *forecast* weather) with what really happened.

Source: MET Norway Locationforecast, CC BY 4.0 / NLOD - no key, free for any
use, attribution required. Their terms require an identifying User-Agent.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from build_dataset import CITIES, LEVEL_DAYS, LEVEL_GAP
from download_knmi import DAY_END, DAY_START, RAIN_MM, TZ

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "data" / "processed" / "sessions_clean.csv"
MODEL = ROOT / "models" / "poisson_B.joblib"
RAW_DIR = ROOT / "data" / "forecasts" / "raw"
WEATHER_OUT = ROOT / "data" / "forecasts" / "weather_forecast_daily.csv"
PRED_OUT = ROOT / "data" / "predictions" / "predictions.csv"
FIG = ROOT / "reports" / "figures" / "forecast_latest.png"

URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USER_AGENT = "parking-pressure-weather-prediction/1.0 https://github.com/Claudia29a/parking-pressure-weather-prediction"
DAYS_AHEAD = 7
MIN_DAY_HOURS = 10   # need at least this many of the 14 daytime hours
RESULTS = ROOT / "reports" / "model_results.csv"   # test MAE per city, from train.py

RAINY, DRY = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fetch_forecast(lat: float, lon: float) -> dict:
    # MET Norway rejects coordinates with more than 4 decimals.
    r = requests.get(URL, params={"lat": round(lat, 4), "lon": round(lon, 4)},
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def to_hourly(forecast: dict) -> pd.DataFrame:
    """Hourly rain (mm) and temperature from the mixed 1h/6h MET Norway steps."""
    ts = forecast["properties"]["timeseries"]
    temp = pd.Series({pd.Timestamp(t["time"]): t["data"]["instant"]["details"]["air_temperature"]
                      for t in ts})
    hours = pd.date_range(temp.index.min(), temp.index.max(), freq="h")
    rain = pd.Series(np.nan, index=hours)
    # Hourly steps first (first ~2.5 days) ...
    for t in ts:
        if "next_1_hours" in t["data"]:
            rain[pd.Timestamp(t["time"])] = t["data"]["next_1_hours"]["details"]["precipitation_amount"]
    # ... then 6-hour blocks for later days, spread evenly over their 6 hours.
    for t in ts:
        if "next_6_hours" in t["data"]:
            block = pd.date_range(pd.Timestamp(t["time"]), periods=6, freq="h").intersection(hours)
            amount = t["data"]["next_6_hours"]["details"]["precipitation_amount"]
            gaps = block[rain[block].isna()]
            rain[gaps] = amount / 6
    return pd.DataFrame({"rain_mm": rain,
                         "temp": temp.reindex(hours).interpolate(method="time")})


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Same daytime definition as the KNMI training data."""
    local = hourly.index.tz_convert(TZ)
    day = hourly[(local.hour >= DAY_START) & (local.hour < DAY_END)].dropna()
    day_local = day.index.tz_convert(TZ)
    daily = day.groupby(day_local.date).agg(rain_mm=("rain_mm", "sum"), temp=("temp", "mean"),
                                            hours=("temp", "size"))
    daily = daily[daily["hours"] >= MIN_DAY_HOURS]
    daily["rain"] = (daily["rain_mm"] >= RAIN_MM).astype(int)
    daily.index = pd.to_datetime(daily.index)
    return daily.drop(columns="hours")


def usage_level(daily_counts: pd.Series, day: pd.Timestamp) -> tuple[float, bool]:
    """Avg sessions/day over 28 days ending 7 days before `day`.

    If the parking export doesn't reach that far, the window ends at the last
    day in the export instead (flagged as stale).
    """
    end = day - pd.Timedelta(days=LEVEL_GAP)
    last = daily_counts.index.max()
    stale = end > last
    end = min(end, last)
    window = daily_counts.loc[end - pd.Timedelta(days=LEVEL_DAYS - 1):end]
    return window.mean(), stale


def draw(pred: pd.DataFrame, run_at: pd.Timestamp) -> None:
    mae = pd.read_csv(RESULTS).set_index("model").loc["Poisson B"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, city in zip(axes.flat, CITIES):
        p = pred[pred["city"] == city]
        # Each city its own scale: Tilburg would be invisible next to Amsterdam.
        top = p["predicted"].max() * 1.3
        x = np.arange(len(p))
        colors = [RAINY if r else DRY for r in p["rain"]]
        bars = ax.bar(x, p["predicted"], 0.62, color=colors)
        for bar, v in zip(bars, p["predicted"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.015, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        labels = [f"{d:%a %d}\n{t:.0f} °C" for d, t in zip(p["date"], p["temp"])]
        ax.set_xticks(x, labels, fontsize=8.5, color=INK)
        ax.set_facecolor(SURFACE)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, top)
        ax.set_title(city, loc="left", color=INK, fontsize=12, fontweight="bold")
        ax.set_title(f"typically off by ±{mae[f'MAE {city}']:.1f}/day", loc="right",
                     color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted sessions", color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DRY), plt.Rectangle((0, 0), 1, 1, color=RAINY)]
    fig.legend(handles, ["Dry day", "Rainy day (≥ 1 mm forecast)"], loc="lower left",
               ncols=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(f"Predicted parking sessions per day, next {DAYS_AHEAD} days   ·   "
                 f"made {run_at:%d %b %Y %H:%M}   ·   note: each city has its own scale",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.99, 0.01, "Weather: MET Norway (CC BY 4.0)", ha="right", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160, facecolor=SURFACE)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    run_at = pd.Timestamp.now(tz=TZ).floor("min")
    model = joblib.load(MODEL)

    sessions = pd.read_csv(SESSIONS, parse_dates=["started_at"])
    sessions = sessions[sessions["city"].isin(CITIES)]
    city_loc = sessions.groupby("city")[["selling_point_lat", "selling_point_lng"]].median()
    sessions["date"] = sessions["started_at"].dt.normalize()

    tomorrow = run_at.tz_localize(None).normalize() + pd.Timedelta(days=1)
    target_days = pd.date_range(tomorrow, periods=DAYS_AHEAD, freq="D")

    weather_rows, pred_rows = [], []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        lat, lon = city_loc.loc[city]
        forecast = fetch_forecast(lat, lon)
        raw_path = RAW_DIR / f"{run_at:%Y-%m-%dT%H%M}_{city.replace(' ', '_')}.json"
        raw_path.write_text(json.dumps(forecast))

        daily = to_daily(to_hourly(forecast)).reindex(target_days)
        weather_rows.append(daily.assign(city=city))

        counts = sessions[sessions["city"] == city].groupby("date").size()
        counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max()), fill_value=0)
        for day in target_days:
            level, stale = usage_level(counts, day)
            pred_rows.append({"city": city, "date": day, "weekday": day.dayofweek, "level": level,
                              "level_stale": stale, "rain": daily.loc[day, "rain"],
                              "temp": daily.loc[day, "temp"], "rain_mm": daily.loc[day, "rain_mm"]})

    pred = pd.DataFrame(pred_rows)
    missing = pred["temp"].isna()
    if missing.any():
        print(f"no full forecast for {missing.sum()} city-days, skipped")
        pred = pred[~missing]
    pred["rain"] = pred["rain"].astype(int)
    pred["predicted"] = model.predict(pred[["city", "weekday", "level", "rain", "temp"]])
    pred.insert(0, "run_at", run_at)

    weather = pd.concat(weather_rows).rename_axis("date").reset_index()
    weather.insert(0, "run_at", run_at)
    append_csv(weather[["run_at", "city", "date", "rain", "temp", "rain_mm"]].round({"temp": 2, "rain_mm": 2}), WEATHER_OUT)
    append_csv(pred.round({"level": 3, "temp": 2, "rain_mm": 2, "predicted": 3}), PRED_OUT)
    draw(pred, run_at)

    if pred["level_stale"].any():
        last = sessions["date"].max().date()
        print(f"NOTE: parking export ends {last}; usage level for later days uses the last "
              f"{LEVEL_DAYS} days of the export. A fresh export makes this exact.")
    print(pred[["city", "date", "rain", "temp", "level", "predicted"]]
          .assign(date=pred["date"].dt.strftime("%a %d %b")).round(2).to_string(index=False))
    print(f"\nsaved predictions -> {PRED_OUT.relative_to(ROOT)}")
    print(f"saved chart       -> {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

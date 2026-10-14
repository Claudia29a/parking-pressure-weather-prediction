"""Step 3a: build the training table (one row per city per day).

Input:  data/processed/sessions_clean.csv
        data/processed/weather_daily.csv
Output: data/processed/train_daily.csv

Columns:
  city, date, sessions        target: parking sessions started that day
  weekday                     0 = Monday ... 6 = Sunday
  level                       usage level: avg sessions/day in the 4 weeks that
                              ended a week before this day (see below)
  rain, temp                  weather from KNMI

Why `level` instead of a day counter: SimplePark grows fast, so the model must
know how busy "normal" currently is. A day counter only works if growth keeps
the same shape, and tree models can't extrapolate it at all. The recent average
can always be computed, also when forecasting: it ends 7 days back, so it is
known for every day of a 7-day forecast.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "data" / "processed" / "sessions_clean.csv"
WEATHER = ROOT / "data" / "processed" / "weather_daily.csv"
OUT = ROOT / "data" / "processed" / "train_daily.csv"

CITIES = ["Amsterdam", "Rotterdam", "Den Haag", "Tilburg"]
LEVEL_DAYS = 28   # length of the window for the usage level
LEVEL_GAP = 7     # window ends this many days before the predicted day


def main() -> None:
    sessions = pd.read_csv(SESSIONS, parse_dates=["started_at"])
    sessions = sessions[sessions["city"].isin(CITIES)]
    sessions["date"] = sessions["started_at"].dt.normalize()
    weather = pd.read_csv(WEATHER, parse_dates=["date"])
    last_day = weather["date"].max()   # KNMI lags ~2 days behind

    counts = sessions.groupby(["city", "date"]).size()

    rows = []
    for city in CITIES:
        # Every day from the city's first session on, with 0 on days without
        # parking: quiet days are information too.
        days = pd.date_range(counts[city].index.min(), last_day, freq="D")
        c = counts[city].reindex(days, fill_value=0)
        level = c.shift(LEVEL_GAP).rolling(LEVEL_DAYS).mean()
        rows.append(pd.DataFrame({"city": city, "date": days,
                                  "sessions": c.values, "level": level.values}))
    df = pd.concat(rows, ignore_index=True)

    df = df.dropna(subset=["level"])   # first 5 weeks per city have no level yet
    df["weekday"] = df["date"].dt.dayofweek
    df = df.merge(weather[["city", "date", "rain", "temp"]], on=["city", "date"], how="inner")
    df = df[["city", "date", "weekday", "level", "rain", "temp", "sessions"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT.relative_to(ROOT)}")
    summary = df.groupby("city").agg(first=("date", "min"), last=("date", "max"), days=("date", "size"),
                                     total=("sessions", "sum"), zero_days=("sessions", lambda s: (s == 0).mean()),
                                     mean=("sessions", "mean"))
    print(summary.round({"zero_days": 2, "mean": 2}).to_string())


if __name__ == "__main__":
    main()

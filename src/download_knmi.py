"""Step 2: download historical weather from KNMI (training data).

Input:  data/processed/sessions_clean.csv   (for city locations + period)
Output: data/raw/knmi_hourly.csv            (raw hourly observations)
        data/processed/weather_daily.csv     (city, date, rain, temp, ...)

What it does:
1. Takes the location of each modelled city (median of its parking spots).
2. Picks the nearest KNMI station that actually measures rain + temperature.
3. Downloads hourly rain (RH) and temperature (T) for the whole parking period.
4. Turns that into one row per city per day:
     rain = 1 if >= 1 mm fell during the day (07-21h Dutch time), else 0
     temp = mean temperature during the day (07-21h)

Source: KNMI open data, no API key, free for any use.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "data" / "processed" / "sessions_clean.csv"
HOURLY_OUT = ROOT / "data" / "raw" / "knmi_hourly.csv"
DAILY_OUT = ROOT / "data" / "processed" / "weather_daily.csv"

URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
CITIES = ["Amsterdam", "Rotterdam", "Den Haag", "Tilburg"]
TZ = "Europe/Amsterdam"
DAY_START, DAY_END = 7, 21    # daytime window, local hours [7, 21)
RAIN_MM = 1.0                 # daytime total that makes it a "rainy day"
MIN_COVERAGE = 0.95           # station must have this share of hours filled
CANDIDATES = 4                # nearest stations to try per city


def knmi_request(start: str, end: str, stns: str) -> str:
    """POST to KNMI; start/end as YYYYMMDDHH, stns like '240:344' or 'ALL'."""
    r = requests.post(URL, data={"start": start, "end": end, "vars": "RH:T", "stns": stns}, timeout=120)
    r.raise_for_status()
    return r.text


def station_list() -> pd.DataFrame:
    """Station codes + coordinates, parsed from the header of a tiny request."""
    text = knmi_request("2025010101", "2025010101", "ALL")
    rows = []
    in_block = False
    for line in text.splitlines():
        if line.startswith("# STN") and "LON" in line:
            in_block = True
            continue
        if in_block:
            parts = line.lstrip("# ").split()
            if not parts or not parts[0].isdigit():
                break
            rows.append({"stn": int(parts[0]), "lon": float(parts[1]),
                         "lat": float(parts[2]), "name": " ".join(parts[4:])})
    return pd.DataFrame(rows)


def parse_data(text: str) -> pd.DataFrame:
    """Turn KNMI CSV text into a DataFrame with a UTC timestamp per hour."""
    if "# STN,YYYYMMDD" not in text:
        raise RuntimeError(f"KNMI did not return data:\n{text[:300]}")
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None,
                     names=["stn", "date", "hh", "RH", "T"], skipinitialspace=True,
                     na_values=[""])
    # KNMI hour HH = the hour *ending* at HH UTC (HH runs 1..24), so the hour
    # starts at HH-1 UTC.
    df["time_utc"] = (pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
                      + pd.to_timedelta(df["hh"] - 1, unit="h")).dt.tz_localize("UTC")
    return df.drop(columns=["date", "hh"])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    sessions = pd.read_csv(SESSIONS, parse_dates=["started_at"])
    sessions = sessions[sessions["city"].isin(CITIES)]
    city_loc = sessions.groupby("city")[["selling_point_lat", "selling_point_lng"]].median()
    start = sessions["started_at"].min().normalize() - pd.Timedelta(days=1)
    end = sessions["started_at"].max().normalize() + pd.Timedelta(days=1)
    print(f"period: {start.date()} -> {end.date()}")

    stations = station_list()
    print(f"KNMI stations available: {len(stations)}")

    # Nearest candidate stations per city, then download them all at once.
    picks = {}
    for city, loc in city_loc.iterrows():
        d = haversine_km(loc["selling_point_lat"], loc["selling_point_lng"],
                         stations["lat"], stations["lon"])
        picks[city] = stations.assign(km=d).nsmallest(CANDIDATES, "km")
    codes = sorted({s for p in picks.values() for s in p["stn"]})
    print(f"downloading stations {codes} ...")
    # One request per month: KNMI refuses requests that are too large.
    chunks = []
    for m_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        m_end = min(m_start + pd.offsets.MonthEnd(0), end)
        m_start = max(m_start, start)
        chunks.append(parse_data(knmi_request(m_start.strftime("%Y%m%d01"), m_end.strftime("%Y%m%d24"),
                                              ":".join(map(str, codes)))))
    raw = pd.concat(chunks, ignore_index=True)

    coverage = raw.groupby("stn")[["RH", "T"]].apply(lambda g: g.notna().mean().min())

    hourly = []
    for city, cand in picks.items():
        ok = cand[cand["stn"].map(coverage).fillna(0) >= MIN_COVERAGE]
        best = ok.iloc[0]
        print(f"  {city:<10} -> station {best['stn']} {best['name']:<22} "
              f"{best['km']:.0f} km, coverage {coverage[best['stn']]:.1%}")
        hourly.append(raw[raw["stn"] == best["stn"]].assign(city=city))
    hourly = pd.concat(hourly)

    HOURLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(HOURLY_OUT, index=False)

    # Convert units: RH -1 means a trace (<0.05 mm) -> treat as 0.
    hourly["rain_mm"] = hourly["RH"].clip(lower=0) / 10
    hourly["temp_c"] = hourly["T"] / 10
    local = hourly["time_utc"].dt.tz_convert(TZ)
    hourly["date"] = local.dt.date
    day = hourly[(local.dt.hour >= DAY_START) & (local.dt.hour < DAY_END)]

    daily = (day.groupby(["city", "date"])
                .agg(rain_mm=("rain_mm", "sum"), temp=("temp_c", "mean"), stn=("stn", "first"))
                .reset_index())
    daily["rain"] = (daily["rain_mm"] >= RAIN_MM).astype(int)
    daily["temp"] = daily["temp"].round(1)
    daily = daily[["city", "date", "rain", "temp", "rain_mm", "stn"]]

    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_OUT, index=False)
    print(f"\nsaved {len(daily)} city-days to {DAILY_OUT.relative_to(ROOT)}")
    print(daily.groupby("city").agg(days=("date", "size"), rainy_share=("rain", "mean"),
                                    mean_temp=("temp", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()

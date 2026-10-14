# Parking pressure × weather prediction

School project (Fontys, semester 7 minor) for stakeholder **SimplePark**. Goal: predict parking pressure per city per hour in the Netherlands for the next 7 days, and prove that weather adds predictive value.

The user wants Claude to do the technical work and explain each step briefly. Ask the user only for things Claude cannot get itself (files, credentials, decisions).

## Decisions already made (don't re-litigate)

- **Direction:** prediction, not live availability (stakeholder chose this).
- **Weather features:** only `rain` (yes/no, 0/1) and `temp` (°C). No wind, no rain amounts.
- **Other features:** only `hour`, `weekday`, `city`. No holidays or events for now.
- **Forecast horizon:** 7 days.
- **Granularity: per city per DAY, top 4 cities only** (Amsterdam, Rotterdam, Den Haag, Tilburg). The export is ALL of SimplePark's data (6,862 clean sessions, May 2025 to Oct 2026; Amsterdam about 3.5 sessions/day), which is too sparse for hourly prediction or other cities.
- **Daily weather definition:** `rain = 1` if at least 1 mm falls 07:00–21:00 Europe/Amsterdam, and `temp` = mean temperature 07:00–21:00. Use the same definition for forecasts.
- **Strong usage growth over time** (57 sessions/month to more than 1,000): the model needs a trend feature, or weather and season get confounded with app adoption.
- **Licensing:** sources must need no API key, no access request, and no payment, and must allow commercial use. Open-Meteo is excluded because its free tier is non-commercial only.

## Data sources

### Historical weather (training): KNMI hourly station data
- Make a POST request to `https://www.daggegevens.knmi.nl/klimatologie/uurgegevens` (no key). Params: `start`, `end` (YYYYMMDDHH), `stns` (station codes joined by `:`), `vars=RH:T`, and optionally `fmt=json`.
- `RH` is hourly precipitation in 0.1 mm, where -1 means a trace under 0.05 mm. `T` is temperature in 0.1 °C.
- Convert with `rain = 1 if RH >= 1 else 0` and `temp = T / 10`.
- The KNMI hour `HH` runs 1–24 and means the hour *ending* at HH UTC, so convert it carefully to a timestamp.
- Map each city to its nearest KNMI station in code, using the parking data's `selling_point_lat/lng` and the station coordinates from the response header.
- The data lags about 48 hours behind real time. It is open data with no restrictions.

### Forecast weather (prediction): MET Norway Locationforecast
- `GET https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=..&lon=..` (no key).
- A User-Agent header with an app name and contact is **required**, for example `SimplePark-Fontys-Project/1.0 <contact email>`. Ask the user for the email.
- Use `air_temperature` and `precipitation_amount` from `next_1_hours`, or `next_6_hours` where hourly data is missing. Set `rain = 1 if amount >= 0.1 else 0`, and apply 6-hour blocks to every hour in the block.
- The license is NLOD / CC BY 4.0, so attribution is required. Cache responses and respect `Expires` / `If-Modified-Since`.
- **Archive every forecast download** together with its download timestamp. No free, commercially usable archive of historical Dutch forecasts exists, so we build our own.

## Modelling plan

1. (done) Clean the parking data. Drop `duration_min == 0`, drop broken or malformed rows (the sample's last line was corrupt), and report how many rows were dropped.
2. Build the target: sessions started per city per day (top 4 cities), with zero-filled days.
3. Join the weather on city and hour (UTC vs. Europe/Amsterdam must be handled explicitly).
4. Explore with a chart of average occupancy on rainy vs. dry hours, per hour of day.
5. Split by time (earlier months for training, the last month for testing). **Never split randomly.**
6. Compare the models:
   - **Model A:** hour + weekday + city
   - **Model B:** A + rain + temp
   - Start with Poisson regression (explainable as "rain changes sessions by X%"), then try LightGBM. The metric is MAE ("spots off").
7. The "honest" evaluation: train on KNMI observations, then after a few weeks measure accuracy when the model is fed the archived MET Norway forecasts instead.

## Prediction pipeline (daily script)

1. Fetch the MET Norway forecast for each city and save the raw response to the archive.
2. Build rows for 7 days × 24 hours × each city, with columns city, date, hour, weekday, rain, temp.
3. Load the trained model, run `predict()`, and store the predictions.
4. Show the result as an hourly chart per city with a rain indicator under each hour.

## Status / open items

- [x] Parking CSV is in `data/raw/parking_sessions.csv` (git-ignored, never push `data/`). Cleaned by `src/clean_parking.py`, which outputs `data/processed/sessions_clean.csv`.
- [x] KNMI history: `src/download_knmi.py` writes `data/processed/weather_daily.csv`. Stations: 240 Schiphol (Amsterdam), 344 Rotterdam, 215 Voorschoten (Den Haag), 350 Gilze-Rijen (Tilburg).
- [ ] Contact email for the MET Norway User-Agent.
- [x] Step 3: `src/build_dataset.py` writes `data/processed/train_daily.csv`, and `src/explore.py` writes `reports/figures/weather_vs_parking.png`. The trend feature is `level` = mean sessions/day over 28 days ending 7 days earlier (works for 7-day forecasts and avoids extrapolation problems).
- [x] Step 4: `src/train.py` (test = from 2026-08-17). Result on 2026-10-14: weather gives NO measurable MAE gain (Poisson A = B = 2.55, naive 2.78, LightGBM worse at 3.14). GLM: rain +0.9% (p=0.85, not significant), temp +0.9%/°C (p=0.014). Overdispersion 1.89.
- [x] Step 5: `src/predict.py` (run daily) fetches MET Norway (User-Agent = repo URL), archives the raw JSON in `data/forecasts/raw/`, and appends to `data/forecasts/weather_forecast_daily.csv` + `data/predictions/predictions.csv`. It draws `reports/figures/forecast_latest.png` with Poisson B. The usage level is flagged `level_stale` when the parking export doesn't reach 7 days before the target day, so it needs fresh exports from SimplePark.
- [ ] Step 6: compare predictions with actual sessions once a new parking export arrives (use the latest run per target day).
- Git: the shell has no GitHub credentials, so the user pushes from VS Code. Commits carry NO Claude co-author lines.
- [x] `.venv` (Python 3.14) with pandas, numpy, requests, matplotlib, scikit-learn and lightgbm. Run scripts as `.venv/bin/python src/<script>.py`.

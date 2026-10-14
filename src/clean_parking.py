"""Step 1: clean the raw SimplePark session export.

Input:  data/raw/parking_sessions.csv
Output: data/processed/sessions_clean.csv

What it does:
1. Repairs rows the export wrapped in one big quoted string (happens when the
   location description contains a comma) instead of dropping them.
2. Normalizes city names that appear under two spellings (incl. accents).
3. Drops 0-minute sessions (mistakes / cancellations) and rows without a city.
4. Prints a short report of how many rows each step removed.
"""
import csv
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "parking_sessions.csv"
OUT = ROOT / "data" / "processed" / "sessions_clean.csv"

# Same place, different name in the export -> one canonical name.
CITY_ALIASES = {
    "'s-Gravenhage": "Den Haag",
    "'s-Hertogenbosch": "Den Bosch",
    "Ouder Amstel": "Ouder-Amstel",
    "Geldrop": "Geldrop-Mierlo",
    "Zaandam": "Zaanstad",
    "Spijkenisse": "Nissewaard",
    "Bergen (NH)": "Bergen",
}


def read_raw(path: Path) -> pd.DataFrame:
    """Read the CSV, unwrapping lines that were quoted as a whole."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header, fixed, repaired = lines[0], [], 0
    for line in lines[1:]:
        if line.startswith('"') and line.endswith('"'):
            # The whole row is one quoted field: unquote it to get the real row.
            line = next(csv.reader([line]))[0]
            repaired += 1
        fixed.append(line)
    print(f"repaired wrapped rows:   {repaired}")
    return pd.read_csv(io.StringIO("\n".join([header, *fixed])))


def main() -> None:
    df = read_raw(RAW)
    n0 = len(df)
    print(f"rows read:               {n0}")

    for col in ["started_at", "ended_at", "preset_end_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    bad = df["started_at"].isna() | df["ended_at"].isna()
    df = df[~bad]
    print(f"dropped unparseable:     {bad.sum()}")

    # Strip accents too: the export has both "Súdwest-Fryslân" and "Sudwest-Fryslan".
    df["city"] = (df["city"].str.strip().str.normalize("NFKD")
                  .str.encode("ascii", "ignore").str.decode("ascii").replace(CITY_ALIASES))
    no_city = df["city"].isna()
    df = df[~no_city]
    print(f"dropped missing city:    {no_city.sum()}")

    zero = df["duration_min"] == 0
    df = df[~zero]
    print(f"dropped 0-min sessions:  {zero.sum()}")

    df = df.sort_values("started_at").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"rows kept:               {len(df)} ({len(df) / n0:.0%})")
    print(f"cities:                  {df['city'].nunique()}")
    print(f"period:                  {df['started_at'].min()} -> {df['started_at'].max()}")
    print(f"saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

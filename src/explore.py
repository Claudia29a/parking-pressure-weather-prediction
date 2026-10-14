"""Step 3b: first look at whether weather matters.

Input:  data/processed/train_daily.csv
Output: reports/figures/weather_vs_parking.png

Raw averages would be misleading: usage grows month after month, and summer has
both more users and less rain. So every day is compared with that city's
*normal* level around that date (centered 4-week average):
  relative = sessions / normal   (1.0 = an ordinary day, 1.2 = 20% busier)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "train_daily.csv"
FIG = ROOT / "reports" / "figures" / "weather_vs_parking.png"

RAINY, DRY, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TEMP_BINS = [-20, 5, 10, 15, 20, 25, 45]
TEMP_LABELS = ["< 5", "5-10", "10-15", "15-20", "20-25", "≥ 25"]


def relative_busyness(df: pd.DataFrame) -> pd.Series:
    normal = (df.groupby("city")["sessions"]
                .transform(lambda s: s.rolling(28, center=True, min_periods=14).mean()))
    return df["sessions"] / normal


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["city", "date"])
    df["rel"] = relative_busyness(df) * 100
    df = df.dropna(subset=["rel"])

    by_rain = df.groupby(["city", "rain"])["rel"].agg(["mean", "size"]).unstack("rain")
    df["temp_band"] = pd.cut(df["temp"], TEMP_BINS, labels=TEMP_LABELS, right=False)
    by_temp = df.groupby("temp_band", observed=True)["rel"].agg(["mean", "size"])

    print("Busyness vs normal (100 = ordinary day)")
    print(by_rain.round(0).to_string())
    print(by_temp.round(0).to_string())

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURFACE)

    # Left: rainy vs dry per city
    cities = by_rain.index.tolist()
    x = np.arange(len(cities))
    w = 0.36
    for off, rain, color, label in [(-w / 2 - 0.01, 0, DRY, "Dry day"), (w / 2 + 0.01, 1, RAINY, "Rainy day (≥ 1 mm)")]:
        vals = by_rain[("mean", rain)]
        bars = a.bar(x + off, vals, w, color=color, label=label)
        for bar, v, n in zip(bars, vals, by_rain[("size", rain)]):
            a.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
                   va="bottom", fontsize=9, color=INK)
            a.text(bar.get_x() + bar.get_width() / 2, 4, f"n={n}", ha="center",
                   va="bottom", fontsize=7.5, color=SURFACE)
    style(a)
    a.set_xticks(x, cities, color=INK)
    a.set_ylabel("Parking sessions vs. normal (%)", color=INK_2)
    a.set_title("Rainy vs. dry days", loc="left", color=INK, fontsize=12, fontweight="bold")
    a.legend(frameon=False, loc="upper left", ncols=2, labelcolor=INK, bbox_to_anchor=(0, -0.08))

    # Right: by daytime temperature, all 4 cities together
    xs = np.arange(len(by_temp))
    bars = b.bar(xs, by_temp["mean"], 0.62, color=NEUTRAL)
    for bar, v, n in zip(bars, by_temp["mean"], by_temp["size"]):
        b.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
               va="bottom", fontsize=9, color=INK)
        b.text(bar.get_x() + bar.get_width() / 2, 4, f"n={n}", ha="center",
               va="bottom", fontsize=7.5, color=SURFACE)
    style(b)
    b.set_xticks(xs, [f"{l} °C" for l in by_temp.index], color=INK)
    b.set_title("By daytime temperature (4 cities)", loc="left", color=INK, fontsize=12, fontweight="bold")

    top = max(by_rain[("mean", 0)].max(), by_rain[("mean", 1)].max(), by_temp["mean"].max())
    for ax in (a, b):
        ax.set_ylim(0, top * 1.15)
    fig.suptitle("Does weather change parking? (100% = an ordinary day for that city)",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160, facecolor=SURFACE)
    print(f"saved {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

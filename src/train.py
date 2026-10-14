"""Step 4: train the models and prove (or disprove) that weather adds value.

Input:  data/processed/train_daily.csv
Output: models/{poisson,lgbm}_{A,B}.joblib
        reports/model_results.csv

Setup:
  Model A = city + weekday + usage level          (no weather)
  Model B = Model A + rain + temp                 (with weather)
  Each is trained twice: Poisson regression (simple, explainable) and
  LightGBM (many small decision trees, can find non-linear patterns).

  Train on everything before TEST_START, test on the last 8 weeks.
  Score = MAE: average number of sessions per day the prediction is off.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "train_daily.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "model_results.csv"

TEST_START = "2026-08-17"   # last 8 weeks are the test set
FEATURES = {
    "A": ["city", "weekday", "level"],
    "B": ["city", "weekday", "level", "rain", "temp"],
}
BOOTSTRAPS = 2000
rng = np.random.default_rng(42)


def poisson_model(features: list[str]):
    numeric = [f for f in features if f in ("rain", "temp")]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), ["city", "weekday"]),
        # log(level): "twice as busy lately" -> "twice as busy today"
        ("level", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["level"]),
        *([("weather", StandardScaler(), numeric)] if numeric else []),
    ])
    return make_pipeline(prep, PoissonRegressor(alpha=1e-4, max_iter=1000))


def lgbm_model():
    return lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.03,
                             num_leaves=7, min_child_samples=30, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                             random_state=42)


def as_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["city"] = X["city"].astype("category")
    return X


def bootstrap_gain(y, pred_a, pred_b) -> tuple[float, float]:
    """95% interval for 'MAE of A minus MAE of B', resampling test days."""
    err_a, err_b = np.abs(y - pred_a), np.abs(y - pred_b)
    idx = rng.integers(0, len(y), (BOOTSTRAPS, len(y)))
    diffs = err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def explain_weather(df: pd.DataFrame) -> None:
    """Poisson GLM on all data, to read the weather effect as a percentage."""
    X = pd.get_dummies(df[["city", "weekday"]].astype(str), drop_first=True, dtype=float)
    X["log_level"] = np.log1p(df["level"])
    X["rain"] = df["rain"]
    X["temp"] = df["temp"]
    X = sm.add_constant(X)
    glm = sm.GLM(df["sessions"], X, family=sm.families.Poisson())
    fit = glm.fit()
    dispersion = fit.pearson_chi2 / fit.df_resid
    if dispersion > 1.2:   # counts vary more than Poisson assumes -> widen intervals
        fit = glm.fit(scale="X2")
    print(f"\nWeather effect (Poisson GLM, all {len(df)} days; dispersion {dispersion:.2f})")
    for name, label in [("rain", "rainy day"), ("temp", "+1 °C")]:
        lo, hi = fit.conf_int().loc[name]
        print(f"  {label:<10} {np.exp(fit.params[name]) - 1:+6.1%}  "
              f"(95% CI {np.exp(lo) - 1:+.1%} to {np.exp(hi) - 1:+.1%}, p = {fit.pvalues[name]:.3f})")


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    train, test = df[df["date"] < TEST_START], df[df["date"] >= TEST_START]
    print(f"train: {len(train)} city-days ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"test:  {len(test)} city-days ({test['date'].min().date()} -> {test['date'].max().date()})")
    y = test["sessions"].to_numpy()

    preds = {"Naive (recent average)": test["level"].to_numpy()}
    MODELS.mkdir(exist_ok=True)
    for kind in ["Poisson", "LightGBM"]:
        for name, feats in FEATURES.items():
            if kind == "Poisson":
                model = poisson_model(feats).fit(train[feats], train["sessions"])
                p = model.predict(test[feats])
            else:
                model = lgbm_model().fit(as_lgbm(train[feats]), train["sessions"])
                p = model.predict(as_lgbm(test[feats]))
            preds[f"{kind} {name}"] = p
            joblib.dump(model, MODELS / f"{kind.lower()}_{name}.joblib")

    rows = []
    for model, p in preds.items():
        row = {"model": model, "MAE": mean_absolute_error(y, p)}
        for city in test["city"].unique():
            m = (test["city"] == city).to_numpy()
            row[f"MAE {city}"] = mean_absolute_error(y[m], p[m])
        rows.append(row)
    results = pd.DataFrame(rows).round(3)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS, index=False)
    print(f"\nMAE on test set (sessions per city-day; avg actual = {y.mean():.2f})")
    print(results.to_string(index=False))

    print("\nDoes weather help? (MAE A - MAE B; positive = weather improves)")
    for kind in ["Poisson", "LightGBM"]:
        a, b = preds[f"{kind} A"], preds[f"{kind} B"]
        gain = mean_absolute_error(y, a) - mean_absolute_error(y, b)
        lo, hi = bootstrap_gain(y, a, b)
        print(f"  {kind:<9} {gain:+.3f} sessions/day ({gain / mean_absolute_error(y, a):+.1%}), "
              f"95% interval {lo:+.3f} to {hi:+.3f}")

    explain_weather(df)


if __name__ == "__main__":
    main()

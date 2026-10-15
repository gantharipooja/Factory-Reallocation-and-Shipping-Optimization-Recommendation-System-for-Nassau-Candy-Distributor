#!/usr/bin/env python3
"""
Nassau Candy Distributor - Factory-to-Customer Shipping Route Efficiency Analysis
=================================================================================

End-to-end analytics pipeline:
    1. Load & validate raw data
    2. Clean & standardise (dates, geography, text, postal codes)
    3. Engineer features (lead time, factory, route, delay flag)
    4. Aggregate routes (Factory -> Region, Factory -> State) and benchmark them
    5. Analyse geographic bottlenecks and ship-mode trade-offs
    6. Test whether observed differences are statistically meaningful
    7. Export cleaned data, route tables, figures and an auto-generated insights report

Usage
-----
    python nassau_candy_analysis.py --data Nassau_Candy_Distributor.csv
    python nassau_candy_analysis.py --data data.csv --output results --delay-threshold 6

IMPORTANT DATA NOTE
-------------------
In the supplied file every Ship Date lies 904-1,642 days after its Order Date, in
three tight clusters (~904, ~1,269 and ~1,637 days). That is a systematic date
offset (a data-export artefact), not real transit time. The pipeline detects the
clusters automatically and subtracts each cluster's baseline, which recovers a
realistic lead time (Same Day ~0 days, Standard Class ~5 days). Each cluster's baseline is
anchored on its Same Day shipments, whose true lead time is ~0. The raw
value is kept in `raw_lead_time_days`. Use `--no-calibrate` to disable this.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Plotting libraries are imported lazily (see _load_plotting) so that importing this module -
# for example from the Streamlit dashboard - needs only pandas and numpy.
matplotlib = plt = sns = None

logger = logging.getLogger("nassau_candy")

# --------------------------------------------------------------------------- #
# Reference data (from the project brief)
# --------------------------------------------------------------------------- #
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

PRODUCT_TO_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

SHIP_MODE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]

# US tile-grid layout (column, row) used for the dependency-free "map".
US_TILE_GRID = {
    "Maine": (11, 0), "Vermont": (10, 1), "New Hampshire": (11, 1),
    "Washington": (1, 2), "Idaho": (2, 2), "Montana": (3, 2), "North Dakota": (4, 2),
    "Minnesota": (5, 2), "Illinois": (6, 2), "Wisconsin": (7, 2), "Michigan": (8, 2),
    "New York": (9, 2), "Rhode Island": (10, 2), "Massachusetts": (11, 2),
    "Oregon": (1, 3), "Nevada": (2, 3), "Wyoming": (3, 3), "South Dakota": (4, 3),
    "Iowa": (5, 3), "Indiana": (6, 3), "Ohio": (7, 3), "Pennsylvania": (8, 3),
    "New Jersey": (9, 3), "Connecticut": (10, 3),
    "California": (1, 4), "Utah": (2, 4), "Colorado": (3, 4), "Nebraska": (4, 4),
    "Missouri": (5, 4), "Kentucky": (6, 4), "West Virginia": (7, 4), "Virginia": (8, 4),
    "Maryland": (9, 4), "Delaware": (10, 4),
    "Arizona": (2, 5), "New Mexico": (3, 5), "Kansas": (4, 5), "Arkansas": (5, 5),
    "Tennessee": (6, 5), "North Carolina": (7, 5), "South Carolina": (8, 5),
    "District of Columbia": (9, 5),
    "Oklahoma": (4, 6), "Louisiana": (5, 6), "Mississippi": (6, 6), "Alabama": (7, 6),
    "Georgia": (8, 6),
    "Texas": (4, 7), "Florida": (9, 7),
}
STATE_ABBR = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY",
}

REQUIRED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Country/Region", "City", "State/Province", "Postal Code", "Division", "Region",
    "Product ID", "Product Name", "Sales", "Units", "Gross Profit", "Cost",
]


@dataclass
class Config:
    data_path: Path
    output_dir: Path = Path("outputs")
    delay_threshold: int = 6          # days; a shipment is "delayed" if lead time > threshold
    min_route_shipments: int = 30     # minimum volume for a state route to be ranked
    calibrate_dates: bool = True
    drop_duplicates: bool = False     # rows identical except Row ID are flagged, not dropped, by default
    cluster_gap_days: int = 30        # gap that separates date-offset clusters
    max_valid_lead_time: int = 60     # sanity ceiling after calibration
    cleaning_log: dict = field(default_factory=dict)

    @property
    def fig_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"


# --------------------------------------------------------------------------- #
# 1. Load & validate
# --------------------------------------------------------------------------- #
def load_data(cfg: Config) -> pd.DataFrame:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.data_path}")
    df = pd.read_csv(cfg.data_path, dtype={"Postal Code": str})
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    logger.info("Loaded %s rows x %s columns", *df.shape)
    cfg.cleaning_log["rows_raw"] = len(df)
    return df


# --------------------------------------------------------------------------- #
# 2. Clean
# --------------------------------------------------------------------------- #
def _calibrate_lead_time(raw: pd.Series, ship_mode: pd.Series, gap: int,
                         anchor: str = "Same Day", min_anchor: int = 10) -> tuple[pd.Series, dict]:
    """Remove systematic date offsets.

    1. Sort raw lead times and split them into clusters wherever consecutive values jump
       by more than `gap` days.
    2. Within each cluster the baseline is the modal raw lead time of `anchor` shipments
       (Same Day orders ship on the order date, so their true lead time is ~0). Using the
       mode rather than the cluster minimum stops a single stray record from shifting the
       whole cluster. Falls back to the cluster minimum if fewer than `min_anchor`
       anchor rows exist.
    3. Subtract the baseline. Records that become negative are removed later as invalid.

    Returns the calibrated series and the {cluster: baseline} mapping for the audit log.
    """
    cluster = (raw.sort_values().diff() > gap).cumsum().reindex(raw.index)
    is_anchor = ship_mode.eq(anchor).fillna(False).astype(bool)
    baselines = {}
    for cid, idx in raw.groupby(cluster).groups.items():
        anchor_vals = raw.loc[idx][is_anchor.loc[idx]]
        base = anchor_vals.mode().min() if len(anchor_vals) >= min_anchor else raw.loc[idx].min()
        baselines[int(cid)] = int(base)
    return raw - cluster.map(baselines), baselines


def clean_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    log = cfg.cleaning_log
    df = df.copy()

    # Text standardisation
    text_cols = ["Order ID", "Ship Mode", "Country/Region", "City", "State/Province",
                 "Division", "Region", "Product ID", "Product Name"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["Product Name"] = df["Product Name"].str.replace(r"-(?=\S)", "- ", regex=True)  # "Bar -Scrum" -> "Bar - Scrum"
    df["Ship Mode"] = df["Ship Mode"].str.title()
    df["State/Province"] = df["State/Province"].str.title()
    df["Region"] = df["Region"].str.title()

    # Postal codes: restore leading zeros lost in CSV round-trips (US ZIPs are 5 digits)
    pc = df["Postal Code"].astype("string").str.strip()
    is_us = df["Country/Region"].eq("United States")
    df["Postal Code"] = pc.where(~is_us, pc.str.zfill(5))

    # Duplicates: rows identical in every field except Row ID. They may be genuine repeat
    # line items, so they are flagged and only dropped when explicitly requested.
    dup_mask = df.duplicated(subset=[c for c in df.columns if c != "Row ID"], keep="first")
    log["possible_duplicate_rows"] = int(dup_mask.sum())
    if cfg.drop_duplicates:
        df = df.loc[~dup_mask].copy()
        log["duplicates_removed"] = int(dup_mask.sum())
    else:
        df["is_possible_duplicate"] = dup_mask

    # Dates (dataset uses DD-MM-YYYY)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
    bad_dates = df[["Order Date", "Ship Date"]].isna().any(axis=1)
    log["invalid_or_missing_dates_removed"] = int(bad_dates.sum())
    df = df.loc[~bad_dates].copy()

    # Numeric validation
    num_cols = ["Sales", "Units", "Gross Profit", "Cost"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_num = df[num_cols].isna().any(axis=1) | (df["Sales"] <= 0) | (df["Units"] <= 0)
    log["invalid_numeric_rows_removed"] = int(bad_num.sum())
    df = df.loc[~bad_num].copy()

    # Lead time
    df["raw_lead_time_days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    log["raw_lead_time_min"] = int(df["raw_lead_time_days"].min())
    log["raw_lead_time_max"] = int(df["raw_lead_time_days"].max())

    if cfg.calibrate_dates and df["raw_lead_time_days"].median() > cfg.max_valid_lead_time:
        df["lead_time_days"], baselines = _calibrate_lead_time(
            df["raw_lead_time_days"], df["Ship Mode"], cfg.cluster_gap_days)
        log["date_offset_clusters_detected"] = len(baselines)
        log["calibration_baselines_days"] = sorted(baselines.values())
        logger.warning("Raw lead times (%s-%s days) are implausible; calibrated %s offset clusters "
                       "(baselines %s days) using Same Day shipments as the zero anchor.",
                       log["raw_lead_time_min"], log["raw_lead_time_max"], len(baselines),
                       log["calibration_baselines_days"])
    else:
        df["lead_time_days"] = df["raw_lead_time_days"]
        log["date_offset_clusters_detected"] = 0

    # Remove invalid lead times (negative or beyond sanity ceiling)
    invalid_lt = (df["lead_time_days"] < 0) | (df["lead_time_days"] > cfg.max_valid_lead_time)
    log["invalid_lead_time_removed"] = int(invalid_lt.sum())
    df = df.loc[~invalid_lt].copy()

    # Integrity: Gross Profit should equal Sales - Cost
    log["profit_identity_violations"] = int(((df["Sales"] - df["Cost"] - df["Gross Profit"]).abs() > 0.01).sum())

    log["rows_clean"] = len(df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["Factory"] = df["Product Name"].map(PRODUCT_TO_FACTORY)
    unmapped = df["Factory"].isna().sum()
    if unmapped:
        logger.warning("%s rows have no factory mapping and will be dropped", unmapped)
        df = df.dropna(subset=["Factory"])

    df["Ship Mode"] = pd.Categorical(df["Ship Mode"], SHIP_MODE_ORDER, ordered=True)
    df["route_region"] = df["Factory"] + " → " + df["Region"]
    df["route_state"] = df["Factory"] + " → " + df["State/Province"]
    df["is_delayed"] = df["lead_time_days"] > cfg.delay_threshold
    df["order_month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["profit_margin"] = df["Gross Profit"] / df["Sales"]

    # Ship-mode-adjusted lead time: removes the effect of mode mix so that routes
    # are not rewarded merely for using faster (more expensive) services.
    mode_mean = df.groupby("Ship Mode", observed=True)["lead_time_days"].transform("mean")
    df["lead_time_vs_mode"] = df["lead_time_days"] - mode_mean
    return df


# --------------------------------------------------------------------------- #
# 4. Route aggregation & benchmarking
# --------------------------------------------------------------------------- #
def aggregate_routes(df: pd.DataFrame, route_col: str, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)
    out = g.agg(
        shipments=("Order ID", "size"),
        orders=("Order ID", "nunique"),
        avg_lead_time=("lead_time_days", "mean"),
        median_lead_time=("lead_time_days", "median"),
        std_lead_time=("lead_time_days", "std"),
        delay_pct=("is_delayed", lambda s: 100 * s.mean()),
        mode_adj_lead_time=("lead_time_vs_mode", "mean"),
        sales=("Sales", "sum"),
        gross_profit=("Gross Profit", "sum"),
    ).reset_index()
    out["std_lead_time"] = out["std_lead_time"].fillna(0.0)
    out["cv_lead_time"] = out["std_lead_time"] / out["avg_lead_time"].replace(0, np.nan)
    out.insert(0, "route", out[group_cols].astype(str).agg(" → ".join, axis=1))
    return out


def add_efficiency_score(routes: pd.DataFrame) -> pd.DataFrame:
    """Route Efficiency Score: min-max normalised average lead time, 100 = fastest."""
    lo, hi = routes["avg_lead_time"].min(), routes["avg_lead_time"].max()
    span = hi - lo
    routes = routes.copy()
    routes["efficiency_score"] = 100.0 if span == 0 else 100 * (hi - routes["avg_lead_time"]) / span
    routes["rank"] = routes["avg_lead_time"].rank(method="min").astype(int)
    return routes.sort_values(["avg_lead_time", "shipments"], ascending=[True, False]).reset_index(drop=True)


def rank_routes(state_routes: pd.DataFrame, min_shipments: int):
    eligible = add_efficiency_score(state_routes[state_routes["shipments"] >= min_shipments])
    return eligible, eligible.head(10), eligible.tail(10).iloc[::-1]


def geographic_bottlenecks(df: pd.DataFrame, min_shipments: int) -> pd.DataFrame:
    """State-level view with a bottleneck flag: above-median volume AND above-average lead time."""
    states = aggregate_routes(df, "State/Province", ["State/Province"])
    states = states[states["shipments"] >= min_shipments].copy()
    overall = df["lead_time_days"].mean()
    states["vs_national_avg"] = states["avg_lead_time"] - overall
    states["bottleneck"] = (states["shipments"] >= states["shipments"].median()) & \
                           (states["avg_lead_time"] > overall)
    return states.sort_values("avg_lead_time", ascending=False).reset_index(drop=True)


def ship_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("Ship Mode", observed=True).agg(
        shipments=("Order ID", "size"),
        avg_lead_time=("lead_time_days", "mean"),
        median_lead_time=("lead_time_days", "median"),
        std_lead_time=("lead_time_days", "std"),
        delay_pct=("is_delayed", lambda s: 100 * s.mean()),
        avg_sales=("Sales", "mean"),
        avg_gross_profit=("Gross Profit", "mean"),
        avg_margin_pct=("profit_margin", lambda s: 100 * s.mean()),
    ).reset_index()
    return out


def delay_sensitivity(df: pd.DataFrame, thresholds=range(4, 9)) -> pd.DataFrame:
    """Share of shipments exceeding each candidate delay threshold (overall and Standard Class)."""
    std = df["Ship Mode"].eq("Standard Class")
    rows = [{"threshold_days": t,
             "delayed_shipments": int((df["lead_time_days"] > t).sum()),
             "delay_pct_all": 100 * (df["lead_time_days"] > t).mean(),
             "delay_pct_standard": 100 * (df.loc[std, "lead_time_days"] > t).mean()} for t in thresholds]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Statistical testing
# --------------------------------------------------------------------------- #
def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis (non-parametric; lead times are discrete and skewed) + eta-squared-style effect size."""
    from scipy import stats  # imported here so the module imports without SciPy

    results = {}
    for label, col in [("Ship Mode", "Ship Mode"), ("Customer Region", "Region"), ("Factory", "Factory")]:
        groups = [g["lead_time_days"].to_numpy() for _, g in df.groupby(col, observed=True)]
        h, p = stats.kruskal(*groups)
        n, k = len(df), len(groups)
        results[label] = {"H": float(h), "p_value": float(p), "epsilon_squared": float((h - k + 1) / (n - k))}
    return results


def route_spread_permutation_test(df: pd.DataFrame, min_shipments: int,
                                  n_perm: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Is the best-to-worst route gap larger than chance would produce?

    Null hypothesis: route (factory -> state) carries no information about lead time once
    ship mode is accounted for. Lead times are shuffled within each ship mode, keeping route
    sizes fixed, and the gap between the slowest and fastest eligible route is recomputed.
    """
    codes = df["route_state"].astype("category").cat.codes.to_numpy()
    counts = np.bincount(codes)
    eligible = counts >= min_shipments
    lead = df["lead_time_days"].to_numpy(dtype=float)
    mode_groups = [np.flatnonzero(m) for m in
                   (df["Ship Mode"].to_numpy() == k for k in df["Ship Mode"].cat.categories)]

    def spread(values: np.ndarray) -> float:
        means = np.bincount(codes, weights=values, minlength=len(counts))[eligible] / counts[eligible]
        return float(means.max() - means.min())

    observed = spread(lead)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = lead.copy()
        for idx in mode_groups:
            shuffled[idx] = rng.permutation(lead[idx])
        null[i] = spread(shuffled)
    return pd.DataFrame([{
        "ranked_routes": int(eligible.sum()), "observed_spread_days": observed,
        "null_mean_spread_days": float(null.mean()), "null_p95_spread_days": float(np.percentile(null, 95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)), "permutations": n_perm}])


# --------------------------------------------------------------------------- #
# 6. Visualisation
# --------------------------------------------------------------------------- #
PALETTE: list = []


def _load_plotting() -> None:
    """Import matplotlib/seaborn on demand with a headless-safe backend."""
    global matplotlib, plt, sns, PALETTE
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.colors  # noqa: F401
    import matplotlib.pyplot as _plt
    import seaborn as _sns

    matplotlib, plt, sns = _mpl, _plt, _sns
    PALETTE = sns.color_palette("viridis", 5)


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.titleweight": "bold",
                         "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig: plt.Figure, cfg: Config, name: str) -> None:
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_data_quality(df: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["raw_lead_time_days"], bins=60, color="#c0392b", ax=axes[0])
    axes[0].set(title="Raw lead time (Ship − Order date)", xlabel="Days", ylabel="Shipments")
    sns.histplot(df["lead_time_days"], discrete=True, color=PALETTE[2], ax=axes[1])
    axes[1].axvline(cfg.delay_threshold + 0.5, ls="--", c="k", lw=1)
    axes[1].text(cfg.delay_threshold + 0.7, axes[1].get_ylim()[1] * 0.9,
                 f"Delay threshold\n> {cfg.delay_threshold} days", fontsize=9)
    axes[1].set(title="Calibrated lead time", xlabel="Days", ylabel="Shipments")
    fig.suptitle("Data quality: systematic date offset detected and corrected", fontweight="bold")
    _save(fig, cfg, "01_data_quality_lead_time")


def plot_overview(df: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    monthly = df.groupby("order_month").agg(volume=("Order ID", "size"), lead=("lead_time_days", "mean"))
    ax = axes[0, 0]
    ax.bar(monthly.index, monthly["volume"], width=20, color=PALETTE[1], alpha=0.8)
    ax.set(title="Monthly shipment volume & average lead time", ylabel="Shipments")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, monthly["lead"], color="#e67e22", marker="o", ms=4)
    ax2.set_ylabel("Avg lead time (days)", color="#e67e22")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    factory_vol = df["Factory"].value_counts()
    axes[0, 1].barh(factory_vol.index[::-1], factory_vol.values[::-1], color=PALETTE[3])
    axes[0, 1].set(title="Shipments by factory", xlabel="Shipments")
    axes[0, 1].set_xscale("log")

    reg = df.groupby("Region")["Sales"].sum().sort_values()
    axes[1, 0].barh(reg.index, reg.values, color=PALETTE[0])
    axes[1, 0].set(title="Sales by customer region", xlabel="Sales (USD)")

    top_states = df["State/Province"].value_counts().head(10)[::-1]
    axes[1, 1].barh(top_states.index, top_states.values, color=PALETTE[2])
    axes[1, 1].set(title="Top 10 destination states/provinces", xlabel="Shipments")
    _save(fig, cfg, "02_business_overview")


def plot_ship_mode(df: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.boxplot(data=df, x="Ship Mode", y="lead_time_days", hue="Ship Mode", palette="viridis",
                legend=False, ax=axes[0], fliersize=2)
    axes[0].set(title="Lead time by ship mode", xlabel="", ylabel="Days")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(summary["Ship Mode"].astype(str), summary["delay_pct"], color=PALETTE[3])
    axes[1].set(title=f"Delay frequency (> {cfg.delay_threshold} days)", ylabel="% of shipments")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(summary["delay_pct"]):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    sc = axes[2].scatter(summary["avg_lead_time"], summary["avg_margin_pct"],
                         s=summary["shipments"] / 8, c=range(len(summary)), cmap="viridis", alpha=0.85)
    for _, r in summary.iterrows():
        axes[2].annotate(str(r["Ship Mode"]), (r["avg_lead_time"], r["avg_margin_pct"]),
                         xytext=(6, 6), textcoords="offset points", fontsize=9)
    axes[2].set(title="Cost-time trade-off (bubble = volume)", xlabel="Avg lead time (days)",
                ylabel="Avg gross margin (%)")
    _save(fig, cfg, "03_ship_mode_comparison")


def plot_route_rankings(top: pd.DataFrame, bottom: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for ax, data, title, color in [(axes[0], top, "Top 10 most efficient routes", PALETTE[3]),
                                   (axes[1], bottom, "Bottom 10 least efficient routes", "#c0392b")]:
        d = data.iloc[::-1]
        ax.barh(d["route"], d["avg_lead_time"], xerr=d["std_lead_time"], color=color, alpha=0.85,
                error_kw={"elinewidth": 1, "alpha": 0.5})
        for y, (v, n) in enumerate(zip(d["avg_lead_time"], d["shipments"])):
            ax.text(v + 0.05, y, f"{v:.2f}d (n={n})", va="center", fontsize=8)
        ax.set(title=title, xlabel="Average lead time (days, ±1 SD)")
    _save(fig, cfg, "04_route_leaderboard")


def plot_factory_region_heatmap(df: pd.DataFrame, cfg: Config) -> None:
    pivot = df.pivot_table(index="Factory", columns="Region", values="lead_time_days", aggfunc="mean")
    counts = df.pivot_table(index="Factory", columns="Region", values="lead_time_days", aggfunc="size")
    annot = pivot.round(2).astype(str) + "\n(n=" + counts.fillna(0).astype(int).astype(str) + ")"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn_r", ax=ax, linewidths=1,
                cbar_kws={"label": "Avg lead time (days)"})
    ax.set(title="Average lead time: Factory → Customer Region", xlabel="", ylabel="")
    _save(fig, cfg, "05_factory_region_heatmap")


def plot_bottlenecks(states: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#c0392b" if flag else "#2a788e" for flag in states["bottleneck"]]
    ax.scatter(states["shipments"], states["avg_lead_time"], s=70, c=colors, alpha=0.85, edgecolor="white")
    overall_avg = np.average(states["avg_lead_time"], weights=states["shipments"])
    ax.axhline(overall_avg, ls="--", c="grey", lw=1)
    ax.axvline(states["shipments"].median(), ls="--", c="grey", lw=1)
    ax.set_xscale("log")
    for _, r in states.iterrows():
        if r["bottleneck"] or r["shipments"] > states["shipments"].quantile(0.9):
            ax.annotate(r["State/Province"], (r["shipments"], r["avg_lead_time"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(title="Geographic bottlenecks: volume vs. average lead time (red = high volume & slow)",
           xlabel="Shipments (log scale)", ylabel="Average lead time (days)")
    _save(fig, cfg, "06_state_bottlenecks")


def plot_tile_map(states: pd.DataFrame, cfg: Config) -> None:
    """US tile-grid heatmap of average lead time (no GIS dependencies needed)."""
    data = states.set_index("State/Province")["avg_lead_time"]
    vmin, vmax = data.min(), data.max()
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for state, (col, row) in US_TILE_GRID.items():
        val = data.get(state, np.nan)
        face = "#e5e5e5" if np.isnan(val) else cmap((val - vmin) / (vmax - vmin + 1e-9))
        ax.add_patch(plt.Rectangle((col, -row), 0.93, 0.93, color=face, ec="white", lw=1.5))
        r, g, b = matplotlib.colors.to_rgb(face)
        txt = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#222222"  # contrast-aware label
        ax.text(col + 0.46, -row + 0.60, STATE_ABBR[state], ha="center", va="center",
                fontsize=10, fontweight="bold", color=txt)
        if not np.isnan(val):
            ax.text(col + 0.46, -row + 0.28, f"{val:.1f}", ha="center", va="center", fontsize=8, color=txt)
    ax.set_xlim(0.5, 12.2)
    ax.set_ylim(-7.3, 1.2)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Avg lead time (days)")
    ax.set_title("US shipping efficiency heatmap (grey = below minimum volume)")
    _save(fig, cfg, "07_us_lead_time_tile_map")


def generate_figures(df, mode_summary, top, bottom, states, cfg: Config) -> None:
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    _load_plotting()
    _setup_style()
    plot_data_quality(df, cfg)
    plot_overview(df, cfg)
    plot_ship_mode(df, mode_summary, cfg)
    plot_route_rankings(top, bottom, cfg)
    plot_factory_region_heatmap(df, cfg)
    plot_bottlenecks(states, cfg)
    plot_tile_map(states, cfg)
    logger.info("Saved figures to %s", cfg.fig_dir)


# --------------------------------------------------------------------------- #
# 7. Insights report
# --------------------------------------------------------------------------- #
def _sig(p: float) -> str:
    return "significant" if p < 0.05 else "not significant"


def build_insights(df, mode_summary, region_routes, ranked, top, bottom, states, tests, cfg: Config) -> str:
    n = len(df)
    avg, med = df["lead_time_days"].mean(), df["lead_time_days"].median()
    delay = 100 * df["is_delayed"].mean()
    fastest_mode = mode_summary.sort_values("avg_lead_time").iloc[0]
    slowest_mode = mode_summary.sort_values("avg_lead_time").iloc[-1]
    std_row = mode_summary[mode_summary["Ship Mode"] == "Standard Class"].iloc[0]
    bn = states[states["bottleneck"]]
    reg = df.groupby("Region")["lead_time_days"].mean().sort_values()
    spread_routes = ranked["avg_lead_time"].max() - ranked["avg_lead_time"].min()
    mix_adj_spread = ranked["mode_adj_lead_time"].max() - ranked["mode_adj_lead_time"].min()
    factory_share = (df["Factory"].value_counts(normalize=True) * 100).round(1)
    corr = df[["lead_time_days", "Sales", "Units", "Gross Profit"]].corr(method="spearman")["lead_time_days"]

    lines = [
        "# Nassau Candy - Shipping Route Efficiency: Key Insights",
        "",
        "## Data & method",
        f"- {n:,} shipments analysed after cleaning ({cfg.cleaning_log['rows_raw']:,} raw rows).",
        f"- Raw Ship-minus-Order gaps ran {cfg.cleaning_log['raw_lead_time_min']:,}-"
        f"{cfg.cleaning_log['raw_lead_time_max']:,} days (systematic offset in "
        f"{cfg.cleaning_log['date_offset_clusters_detected']} clusters, baselines "
        f"{cfg.cleaning_log.get('calibration_baselines_days', [])} days, anchored on Same Day shipments); "
        f"{cfg.cleaning_log['invalid_lead_time_removed']} records with negative calibrated lead time removed.",
        f"- Delay = lead time > {cfg.delay_threshold} days. Routes ranked only with ≥ {cfg.min_route_shipments} shipments.",
        "",
        "## Headline KPIs",
        f"- Average lead time **{avg:.2f} days** (median {med:.0f}); delay frequency **{delay:.1f}%**.",
        f"- Fastest mode: **{fastest_mode['Ship Mode']}** ({fastest_mode['avg_lead_time']:.2f} days); "
        f"slowest: **{slowest_mode['Ship Mode']}** ({slowest_mode['avg_lead_time']:.2f} days).",
        f"- Standard Class carries {100 * std_row['shipments'] / n:.0f}% of volume and "
        f"{std_row['delay_pct']:.1f}% of its shipments are delayed.",
        "",
        "## Route efficiency",
        f"- Most efficient route: **{top.iloc[0]['route']}** ({top.iloc[0]['avg_lead_time']:.2f} days, "
        f"n={top.iloc[0]['shipments']}).",
        f"- Least efficient route: **{bottom.iloc[0]['route']}** ({bottom.iloc[0]['avg_lead_time']:.2f} days, "
        f"n={bottom.iloc[0]['shipments']}).",
        f"- Spread between best and worst ranked route is only {spread_routes:.2f} days "
        f"({mix_adj_spread:.2f} days after adjusting for ship-mode mix).",
        f"- Customer-region averages range from {reg.iloc[0]:.2f} ({reg.index[0]}) to "
        f"{reg.iloc[-1]:.2f} days ({reg.index[-1]}).",
        "",
        "## Geographic bottlenecks",
    ]
    if len(bn):
        lines.append("- High-volume, slower-than-average states: " + ", ".join(
            f"**{r['State/Province']}** ({r['avg_lead_time']:.2f}d, n={r['shipments']})"
            for _, r in bn.sort_values("shipments", ascending=False).head(6).iterrows()) + ".")
    slow = states.head(3)
    lines.append("- Slowest states overall (min volume applied): " + ", ".join(
        f"{r['State/Province']} ({r['avg_lead_time']:.2f}d)" for _, r in slow.iterrows()) + ".")
    lines += [
        "",
        "## Statistical tests (Kruskal-Wallis on lead time)",
    ]
    for k, v in tests.items():
        lines.append(f"- {k}: H={v['H']:.1f}, p={'<1e-300' if v['p_value'] < 1e-300 else format(v['p_value'], '.3g')} ({_sig(v['p_value'])}), "
                     f"effect size ε²={v['epsilon_squared']:.3f}")
    lines += [
        "",
        "## Interpretation",
        "- Ship mode is the dominant driver of lead time. Region and factory differences are statistically "
        "detectable (large sample) but practically negligible (ε² ≈ 0.001), so state/route gaps of a fraction "
        "of a day should be treated as noise-level variation rather than structural bottlenecks.",
        f"- Volume is concentrated: {factory_share.iloc[0]}% of shipments originate at {factory_share.index[0]}.",
        f"- Lead time shows negligible rank correlation with order value (ρ={corr['Sales']:.2f}) "
        f"and units (ρ={corr['Units']:.2f}).",
        "",
        "## Recommendations",
        "1. Audit Standard Class on the slowest high-volume states; consider upgrading only those lanes to "
        "Second Class where margin allows (see cost-time chart).",
        "2. Track Delay Frequency and the Efficiency Score monthly per route; alert when a route moves > 1 SD.",
        "3. Factory assignment is fixed by product; once real delivery data exists, test whether serving "
        "high-volume demand clusters from a nearer factory shortens lead time (not testable with this data).",
        "4. Capture real shipping cost and carrier/actual-delivery dates - the current data cannot quantify "
        "the cost-time trade-off beyond gross margin.",
        "5. Fix the Ship Date export so lead times are recorded correctly at source.",
        "",
        "## Limitations",
        "- Lead time is Order → Ship (dispatch), not delivery. No carrier cost data; factory coordinates are not "
        "used for distance because customer coordinates are absent.",
        "- Canadian provinces are ranked but not drawn on the US tile map.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    cfg.table_dir.mkdir(parents=True, exist_ok=True)
    raw = load_data(cfg)
    df = engineer_features(clean_data(raw, cfg), cfg)
    logger.info("Cleaning summary: %s", cfg.cleaning_log)

    region_routes = add_efficiency_score(aggregate_routes(df, "route_region", ["Factory", "Region"]))
    state_routes = aggregate_routes(df, "route_state", ["Factory", "State/Province"])
    ranked, top10, bottom10 = rank_routes(state_routes, cfg.min_route_shipments)
    states = geographic_bottlenecks(df, cfg.min_route_shipments)
    mode_summary = ship_mode_summary(df)
    mode_route = aggregate_routes(df, "route_region", ["Factory", "Region", "Ship Mode"])
    tests = run_statistical_tests(df)

    # Exports
    df.to_csv(cfg.output_dir / "nassau_candy_clean.csv", index=False)
    region_routes.to_csv(cfg.table_dir / "routes_factory_region.csv", index=False)
    ranked.to_csv(cfg.table_dir / "routes_factory_state_ranked.csv", index=False)
    top10.to_csv(cfg.table_dir / "top10_routes.csv", index=False)
    bottom10.to_csv(cfg.table_dir / "bottom10_routes.csv", index=False)
    states.to_csv(cfg.table_dir / "state_bottlenecks.csv", index=False)
    mode_summary.to_csv(cfg.table_dir / "ship_mode_summary.csv", index=False)
    delay_sensitivity(df).to_csv(cfg.table_dir / "delay_threshold_sensitivity.csv", index=False)
    perm = route_spread_permutation_test(df, cfg.min_route_shipments)
    perm.to_csv(cfg.table_dir / "route_spread_permutation_test.csv", index=False)
    logger.info("Route-spread permutation test: %s", perm.round(3).to_dict("records")[0])
    mode_route.to_csv(cfg.table_dir / "routes_by_ship_mode.csv", index=False)

    generate_figures(df, mode_summary, top10, bottom10, states, cfg)

    report = build_insights(df, mode_summary, region_routes, ranked, top10, bottom10, states, tests, cfg)
    (cfg.output_dir / "insights.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info("Done. Outputs written to %s", cfg.output_dir.resolve())


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Nassau Candy shipping route efficiency analysis")
    p.add_argument("--data", required=True, type=Path, help="Path to the CSV dataset")
    p.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    p.add_argument("--delay-threshold", type=int, default=6, help="Delay threshold in days (default 6)")
    p.add_argument("--min-shipments", type=int, default=30, help="Min shipments to rank a state route")
    p.add_argument("--drop-duplicates", action="store_true",
                   help="Drop rows identical in every field except Row ID (flagged by default)")
    p.add_argument("--no-calibrate", action="store_true", help="Disable ship-date offset calibration")
    a = p.parse_args(argv)
    return Config(data_path=a.data, output_dir=a.output, delay_threshold=a.delay_threshold,
                  min_route_shipments=a.min_shipments, calibrate_dates=not a.no_calibrate,
                  drop_duplicates=a.drop_duplicates)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

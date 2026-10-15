"""
Nassau Candy Distributor - Shipping Route Efficiency Dashboard (Streamlit)
==========================================================================

Run from the folder that contains this file, nassau_candy_analysis.py and the CSV:

    pip install -r requirements.txt
    streamlit run app.py

The dashboard reuses the cleaning, calibration and route-aggregation code from
`nassau_candy_analysis.py`, so the numbers match the research paper exactly.

Modules (tabs)
    1. Route Efficiency Overview  - leaderboard and average lead time by route
    2. Geographic Shipping Map    - US heatmap and bottleneck view
    3. Ship Mode Comparison       - lead time, delay and cost-time trade-off
    4. Route Drill-Down           - state insights and order-level timelines

Sidebar filters: date range, region, state, ship mode, factory, delay-threshold slider.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nassau_candy_analysis import (
    FACTORY_COORDS,
    REQUIRED_COLUMNS,
    SHIP_MODE_ORDER,
    STATE_ABBR,
    Config,
    add_efficiency_score,
    aggregate_routes,
    clean_data,
    engineer_features,
)

DEFAULT_CSV = Path(__file__).resolve().parent / "Nassau_Candy_Distributor.csv"
ACCENT = "#1F3A5F"
LEAD_SCALE = "RdYlGn_r"  # green = fast, red = slow

st.set_page_config(page_title="Nassau Candy | Shipping Route Efficiency", page_icon="🍬", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      div[data-testid="stMetric"] {background:#F4F6FA; border-radius:10px; padding:10px 14px;}
      div[data-testid="stMetricValue"] {color:#1F3A5F;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cleaning and calibrating data...")
def prepare(file_bytes: bytes | None, path_str: str) -> tuple[pd.DataFrame, dict]:
    """Load, validate, clean and feature-engineer the dataset."""
    source = io.BytesIO(file_bytes) if file_bytes is not None else path_str
    raw = pd.read_csv(source, dtype={"Postal Code": str})
    raw.columns = raw.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    cfg = Config(data_path=Path(path_str))
    cfg.cleaning_log["rows_raw"] = len(raw)
    df = engineer_features(clean_data(raw, cfg), cfg)
    return df, dict(cfg.cleaning_log)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def route_table(df: pd.DataFrame, level: str, min_shipments: int) -> pd.DataFrame:
    """Aggregate routes at 'Factory → State' or 'Factory → Region' level, rank them, add uncertainty."""
    cols = ["Factory", "State/Province"] if level == "Factory → State" else ["Factory", "Region"]
    routes = aggregate_routes(df, "route", cols)
    routes = routes[routes["shipments"] >= min_shipments]
    if routes.empty:
        return routes
    routes = add_efficiency_score(routes)
    routes["margin_of_error_95"] = 1.96 * routes["std_lead_time"] / np.sqrt(routes["shipments"])
    return routes


def state_table(df: pd.DataFrame, min_shipments: int) -> pd.DataFrame:
    states = aggregate_routes(df, "state", ["State/Province"])
    states = states[states["shipments"] >= min_shipments].copy()
    if states.empty:
        return states
    national = df["lead_time_days"].mean()
    states["vs_national_avg"] = states["avg_lead_time"] - national
    states["bottleneck"] = (states["shipments"] >= states["shipments"].median()) & (states["avg_lead_time"] > national)
    states["abbr"] = states["State/Province"].map(STATE_ABBR)
    return states.sort_values("avg_lead_time", ascending=False).reset_index(drop=True)


def mode_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Ship Mode", observed=True)
    out = g.agg(
        shipments=("Order ID", "size"),
        avg_lead_time=("lead_time_days", "mean"),
        median_lead_time=("lead_time_days", "median"),
        std_lead_time=("lead_time_days", "std"),
        delay_pct=("is_delayed", lambda s: 100 * s.mean()),
        avg_margin_pct=("profit_margin", lambda s: 100 * s.mean()),
        avg_sales=("Sales", "mean"),
    ).reset_index()
    out["Ship Mode"] = out["Ship Mode"].astype(str)
    return out


def show_table(df: pd.DataFrame, rename: dict[str, str], decimals: int = 2) -> None:
    out = df[list(rename)].rename(columns=rename).round(decimals)
    st.dataframe(out, use_container_width=True, hide_index=True)


def fmt_days(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2f} days"


# --------------------------------------------------------------------------- #
# Sidebar: data source and filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory-to-customer shipping route efficiency")

uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type="csv")
if uploaded is None and not DEFAULT_CSV.exists():
    st.title("Shipping Route Efficiency Dashboard")
    st.info(f"Upload the dataset in the sidebar, or place **{DEFAULT_CSV.name}** next to `app.py`.")
    st.stop()

try:
    data, clean_log = prepare(uploaded.getvalue() if uploaded is not None else None, str(DEFAULT_CSV))
except (ValueError, KeyError, FileNotFoundError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.sidebar.header("Filters")
min_d, max_d = data["Order Date"].min().date(), data["Order Date"].max().date()
picked = st.sidebar.date_input("Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = (picked[0], picked[1]) if isinstance(picked, (tuple, list)) and len(picked) == 2 else (min_d, max_d)

regions = sorted(data["Region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Customer region", regions, default=regions)
state_options = sorted(data.loc[data["Region"].isin(sel_regions), "State/Province"].dropna().unique())
sel_states = st.sidebar.multiselect("State / province (empty = all)", state_options, default=[])
sel_modes = st.sidebar.multiselect("Ship mode", SHIP_MODE_ORDER, default=SHIP_MODE_ORDER)
factories = sorted(data["Factory"].unique())
sel_factories = st.sidebar.multiselect("Factory", factories, default=factories)

threshold = st.sidebar.slider(
    "Delay threshold (days)", min_value=1, max_value=10, value=6,
    help="A shipment counts as delayed when its lead time exceeds this many days.",
)
min_ship = st.sidebar.slider(
    "Minimum shipments to rank a route/state", min_value=1, max_value=100, value=30,
    help="Averages from small samples are noisy; raise this to see only well-supported routes.",
)

mask = (
    (data["Order Date"].dt.date >= start_d)
    & (data["Order Date"].dt.date <= end_d)
    & data["Region"].isin(sel_regions)
    & data["Ship Mode"].astype(str).isin(sel_modes)
    & data["Factory"].isin(sel_factories)
)
if sel_states:
    mask &= data["State/Province"].isin(sel_states)
df = data.loc[mask].copy()
df["is_delayed"] = df["lead_time_days"] > threshold

with st.sidebar.expander("About the data"):
    st.markdown(
        f"""
        - **Lead time** = ship date minus order date (order-to-dispatch, not delivery).
        - The raw ship dates carry a systematic offset. It was corrected in
          {clean_log.get('date_offset_clusters_detected', 0)} clusters
          (baselines {clean_log.get('calibration_baselines_days', [])} days) using Same Day
          shipments as the zero anchor.
        - {clean_log.get('invalid_lead_time_removed', 0)} records with negative calibrated lead time were removed.
        - {clean_log.get('possible_duplicate_rows', 0)} rows identical except Row ID are kept (flagged).
        """
    )

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Shipping Route Efficiency Dashboard")
st.caption(f"{start_d:%d %b %Y} to {end_d:%d %b %Y} | delay = lead time above {threshold} days")

if df.empty:
    st.warning("No shipments match the current filters. Widen the date range or selections.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Shipments", f"{len(df):,}")
c2.metric("Avg lead time", fmt_days(df["lead_time_days"].mean()))
c3.metric("Median lead time", f"{df['lead_time_days'].median():.0f} days")
c4.metric("Delay frequency", f"{100 * df['is_delayed'].mean():.1f}%")
c5.metric("Avg gross margin", f"{100 * df['Gross Profit'].sum() / df['Sales'].sum():.1f}%")

tab_overview, tab_map, tab_mode, tab_drill = st.tabs(
    ["Route Efficiency Overview", "Geographic Shipping Map", "Ship Mode Comparison", "Route Drill-Down"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Route Efficiency Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    level = st.radio("Route level", ["Factory → State", "Factory → Region"], horizontal=True)
    routes = route_table(df, level, min_ship)

    if routes.empty:
        st.info("No routes meet the minimum-shipments setting. Lower it in the sidebar.")
    else:
        spread = routes["avg_lead_time"].max() - routes["avg_lead_time"].min()
        moe = routes["margin_of_error_95"].median()
        st.info(
            f"**Read rankings with care.** {len(routes)} routes are ranked. The gap between the fastest and slowest is "
            f"**{spread:.2f} days**, while a typical route's average carries a 95% margin of error of about "
            f"**±{moe:.2f} days**. Differences of that size are often sampling noise, not real route quality."
        )

        n_show = st.slider("Routes to show in each list", 5, 15, 10)
        fastest = routes.head(n_show)
        slowest = routes.tail(n_show).iloc[::-1]

        left, right = st.columns(2)
        for col, data_, title, color in [
            (left, fastest, f"Top {n_show} most efficient routes", "#2E8B57"),
            (right, slowest, f"Bottom {n_show} least efficient routes", "#C0392B"),
        ]:
            fig = go.Figure(
                go.Bar(
                    x=data_["avg_lead_time"], y=data_["route"], orientation="h", marker_color=color,
                    error_x=dict(type="data", array=data_["margin_of_error_95"], visible=True),
                    customdata=np.stack([data_["shipments"], data_["delay_pct"]], axis=-1),
                    hovertemplate="%{y}<br>Avg lead time: %{x:.2f} days<br>Shipments: %{customdata[0]}"
                                  "<br>Delayed: %{customdata[1]:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(title=title, height=420, margin=dict(l=0, r=10, t=40, b=0),
                              xaxis_title="Average lead time (days, ±95% margin of error)",
                              yaxis=dict(autorange="reversed"))
            col.plotly_chart(fig, use_container_width=True)

        st.subheader("Route performance leaderboard")
        show_table(
            routes,
            {"rank": "Rank", "route": "Route", "shipments": "Shipments", "avg_lead_time": "Avg lead time",
             "std_lead_time": "Std dev", "margin_of_error_95": "±95% MoE", "delay_pct": "Delayed %",
             "efficiency_score": "Efficiency score"},
        )
        st.download_button("Download leaderboard (CSV)", routes.to_csv(index=False).encode("utf-8"),
                           file_name="route_leaderboard.csv", mime="text/csv")

        if level == "Factory → Region":
            pivot = routes.pivot_table(index="Factory", columns="Region", values="avg_lead_time")
            heat = px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale=LEAD_SCALE,
                             title="Average lead time: factory to customer region")
            heat.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(heat, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2: Geographic Shipping Map
# --------------------------------------------------------------------------- #
with tab_map:
    metric_label = st.selectbox("Map metric", ["Average lead time (days)", "Delay frequency (%)", "Shipments"])
    metric_col = {"Average lead time (days)": "avg_lead_time", "Delay frequency (%)": "delay_pct",
                  "Shipments": "shipments"}[metric_label]
    states = state_table(df, min_ship)
    us_states = states.dropna(subset=["abbr"]) if not states.empty else states

    if us_states.empty:
        st.info("No US states meet the minimum-shipments setting. Lower it in the sidebar.")
    else:
        fig = px.choropleth(
            us_states, locations="abbr", locationmode="USA-states", scope="usa", color=metric_col,
            color_continuous_scale=LEAD_SCALE if metric_col != "shipments" else "Blues",
            hover_name="State/Province",
            hover_data={"abbr": False, "shipments": True, "avg_lead_time": ":.2f", "delay_pct": ":.1f"},
            labels={metric_col: metric_label},
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[v[1] for v in FACTORY_COORDS.values()], lat=[v[0] for v in FACTORY_COORDS.values()],
                text=list(FACTORY_COORDS), mode="markers", name="Factories",
                marker=dict(size=11, color=ACCENT, symbol="diamond", line=dict(width=1, color="white")),
                hovertemplate="%{text}<extra>Factory</extra>",
            )
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=0.02))
        st.plotly_chart(fig, use_container_width=True)
        canada_n = int((df["Country/Region"] == "Canada").sum())
        if canada_n:
            st.caption(f"{canada_n:,} Canadian shipments are included in all metrics but not drawn on the US map.")

    st.subheader("Regional bottleneck view")
    if states.empty:
        st.info("No states meet the minimum-shipments setting.")
    else:
        national = df["lead_time_days"].mean()
        scatter = px.scatter(
            states, x="shipments", y="avg_lead_time", color="bottleneck", log_x=True,
            color_discrete_map={True: "#C0392B", False: "#2A788E"}, hover_name="State/Province",
            hover_data={"delay_pct": ":.1f", "bottleneck": False},
            labels={"shipments": "Shipments (log scale)", "avg_lead_time": "Average lead time (days)"},
            title="Volume vs. average lead time (red = above-median volume and above-average lead time)",
        )
        scatter.add_hline(y=national, line_dash="dash", line_color="grey",
                          annotation_text=f"Average {national:.2f} days")
        scatter.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(scatter, use_container_width=True)

        flagged = states[states["bottleneck"]].sort_values("shipments", ascending=False)
        if flagged.empty:
            st.success("No states are flagged as bottlenecks under the current filters.")
        else:
            st.markdown(f"**{len(flagged)} states flagged** (above-median volume and above-average lead time):")
            show_table(flagged, {"State/Province": "State", "shipments": "Shipments", "avg_lead_time": "Avg lead time",
                                 "vs_national_avg": "Vs. average (days)", "delay_pct": "Delayed %"})

# --------------------------------------------------------------------------- #
# Tab 3: Ship Mode Comparison
# --------------------------------------------------------------------------- #
with tab_mode:
    modes = mode_table(df)
    left, right = st.columns(2)

    box = px.box(df, x="Ship Mode", y="lead_time_days", color="Ship Mode",
                 category_orders={"Ship Mode": SHIP_MODE_ORDER},
                 labels={"lead_time_days": "Lead time (days)", "Ship Mode": ""},
                 title="Lead time distribution by ship mode")
    box.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    left.plotly_chart(box, use_container_width=True)

    bars = px.bar(modes, x="Ship Mode", y="delay_pct", text=modes["delay_pct"].round(1).astype(str) + "%",
                  category_orders={"Ship Mode": SHIP_MODE_ORDER},
                  labels={"delay_pct": f"Delayed (> {threshold} days), %", "Ship Mode": ""},
                  title=f"Delay frequency (> {threshold} days)")
    bars.update_traces(marker_color=ACCENT)
    bars.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    right.plotly_chart(bars, use_container_width=True)

    st.subheader("Cost-time trade-off (descriptive)")
    tradeoff = px.scatter(modes, x="avg_lead_time", y="avg_margin_pct", size="shipments", text="Ship Mode",
                          labels={"avg_lead_time": "Average lead time (days)", "avg_margin_pct": "Average gross margin (%)"},
                          title="Gross margin vs. lead time (bubble size = shipments)")
    tradeoff.update_traces(textposition="top center", marker=dict(color=ACCENT, opacity=0.8))
    tradeoff.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(tradeoff, use_container_width=True)
    st.caption(
        "Freight cost is not in the dataset, so this chart uses gross margin as a proxy and cannot show what "
        "faster shipping actually costs. Capturing carrier cost per shipment would close this gap."
    )

    show_table(modes, {"Ship Mode": "Ship mode", "shipments": "Shipments", "avg_lead_time": "Avg lead time",
                       "median_lead_time": "Median", "std_lead_time": "Std dev", "delay_pct": "Delayed %",
                       "avg_margin_pct": "Avg margin %", "avg_sales": "Avg sales ($)"})

    pivot = df.pivot_table(index="Ship Mode", columns="Region", values="lead_time_days", aggfunc="mean", observed=True)
    heat = px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale=LEAD_SCALE,
                     title="Average lead time: ship mode by customer region")
    heat.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(heat, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 4: Route Drill-Down
# --------------------------------------------------------------------------- #
with tab_drill:
    available = sorted(df["State/Province"].dropna().unique())
    default_idx = available.index(df["State/Province"].value_counts().idxmax())
    state = st.selectbox("State / province", available, index=default_idx)
    sdf = df[df["State/Province"] == state]
    national = df["lead_time_days"].mean()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Shipments", f"{len(sdf):,}")
    d2.metric("Avg lead time", fmt_days(sdf["lead_time_days"].mean()),
              delta=f"{sdf['lead_time_days'].mean() - national:+.2f} vs. filtered average", delta_color="inverse")
    d3.metric("Delay frequency", f"{100 * sdf['is_delayed'].mean():.1f}%")
    d4.metric("Sales", f"${sdf['Sales'].sum():,.0f}")

    left, right = st.columns(2)
    by_factory = aggregate_routes(sdf, "route", ["Factory"]).sort_values("avg_lead_time")
    fig_f = px.bar(by_factory, x="Factory", y="avg_lead_time", text=by_factory["shipments"].astype(str) + " shipments",
                   labels={"avg_lead_time": "Average lead time (days)"}, title=f"{state}: lead time by origin factory")
    fig_f.update_traces(marker_color=ACCENT)
    fig_f.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
    left.plotly_chart(fig_f, use_container_width=True)

    by_mode = mode_table(sdf)
    fig_m = px.bar(by_mode, x="Ship Mode", y="avg_lead_time", text=by_mode["shipments"].astype(str) + " shipments",
                   category_orders={"Ship Mode": SHIP_MODE_ORDER},
                   labels={"avg_lead_time": "Average lead time (days)"}, title=f"{state}: lead time by ship mode")
    fig_m.update_traces(marker_color="#2A788E")
    fig_m.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
    right.plotly_chart(fig_m, use_container_width=True)

    monthly = sdf.groupby("order_month").agg(avg_lead_time=("lead_time_days", "mean"),
                                             shipments=("Order ID", "size")).reset_index()
    fig_t = px.line(monthly, x="order_month", y="avg_lead_time", markers=True,
                    labels={"order_month": "Order month", "avg_lead_time": "Average lead time (days)"},
                    title=f"{state}: monthly average lead time")
    fig_t.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("Order-level shipment timeline")
    n_orders = st.slider("Most recent orders to show", 10, 100, 40, step=5)
    recent = sdf.sort_values("Order Date", ascending=False).head(n_orders).copy()
    recent["Calibrated ship date"] = recent["Order Date"] + pd.to_timedelta(recent["lead_time_days"], unit="D")
    recent["timeline_end"] = recent["Order Date"] + pd.to_timedelta(np.maximum(recent["lead_time_days"], 0.25), unit="D")
    recent["Order"] = recent["Order ID"] + " #" + recent["Row ID"].astype(str)
    recent["Status"] = np.where(recent["is_delayed"], "Delayed", "On time")
    recent["Ship Mode"] = recent["Ship Mode"].astype(str)

    timeline = px.timeline(
        recent, x_start="Order Date", x_end="timeline_end", y="Order", color="Ship Mode",
        category_orders={"Ship Mode": SHIP_MODE_ORDER}, hover_data={"Product Name": True, "Factory": True,
                                                                     "lead_time_days": True, "Status": True,
                                                                     "timeline_end": False, "Order": False},
    )
    timeline.update_yaxes(autorange="reversed", title="")
    timeline.update_layout(height=max(360, 16 * len(recent) + 120), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(timeline, use_container_width=True)

    order_cols = {"Order ID": "Order ID", "Order Date": "Order date", "Calibrated ship date": "Calibrated ship date",
                  "Ship Mode": "Ship mode", "Factory": "Factory", "Product Name": "Product",
                  "lead_time_days": "Lead time (days)", "Status": "Status", "Sales": "Sales ($)"}
    orders_out = recent[list(order_cols)].rename(columns=order_cols)
    orders_out["Order date"] = orders_out["Order date"].dt.date
    orders_out["Calibrated ship date"] = orders_out["Calibrated ship date"].dt.date
    st.dataframe(orders_out, use_container_width=True, hide_index=True)
    st.download_button("Download these orders (CSV)", orders_out.to_csv(index=False).encode("utf-8"),
                       file_name=f"orders_{state.replace(' ', '_').lower()}.csv", mime="text/csv")

st.caption("Nassau Candy Distributor | Shipping route efficiency analysis | Lead time is order-to-dispatch, calibrated.")

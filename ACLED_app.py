"""Streamlit analytic tool for ACLED data."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "ACLED 2024-2025.csv"
DATE_COL = "event_date"
LAT_COL = "latitude"
LON_COL = "longitude"
COUNTRY_COL = "country"
EVENT_TYPE_COL = "event_type"
SUB_EVENT_COL = "sub_event_type"
ADMIN1_COL = "admin1"
NOTES_COL = "notes"
FATALITIES_COL = "fatalities"
ACTOR1_COL = "actor1"
ASSOC_ACTOR1_COL = "assoc_actor_1"

st.set_page_config(page_title="ACLED Conflict Analytics", layout="wide")

# ---------------------------------------------------------------------------
# Data loading & utilities
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load ACLED data from disk with typed columns."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[FATALITIES_COL] = pd.to_numeric(df[FATALITIES_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, LAT_COL, LON_COL])
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.to_period("M").astype(str)
    df["week"] = df[DATE_COL].dt.to_period("W").astype(str)
    df[NOTES_COL] = df[NOTES_COL].fillna("")
    df[ACTOR1_COL] = df[ACTOR1_COL].fillna("Unknown actor")
    df[ASSOC_ACTOR1_COL] = df[ASSOC_ACTOR1_COL].fillna("")
    return df


def parse_keywords(raw: str) -> List[str]:
    return [kw.strip() for kw in (raw or "").split(",") if kw.strip()]


def filter_by_keywords(
    df: pd.DataFrame,
    keywords: Sequence[str],
    columns: Sequence[str],
    match_all: bool = False,
) -> pd.DataFrame:
    if not keywords:
        return df
    mask = None
    for kw in keywords:
        col_mask = pd.Series(False, index=df.index)
        for col in columns:
            col_mask |= df[col].str.contains(kw, case=False, na=False)
        mask = col_mask if mask is None else (mask & col_mask if match_all else mask | col_mask)
    return df[mask]


def filter_by_context(
    df: pd.DataFrame,
    contexts: Sequence[str],
    match_all: bool = True,
) -> pd.DataFrame:
    if not contexts:
        return df
    mask = pd.Series(True, index=df.index) if match_all else pd.Series(False, index=df.index)
    for ctx in contexts:
        ctx_mask = df[NOTES_COL].str.contains(ctx, case=False, na=False)
        mask = mask & ctx_mask if match_all else mask | ctx_mask
    return df[mask]


def apply_filters(
    df: pd.DataFrame,
    date_range,
    countries: Sequence[str],
    event_types: Sequence[str],
    admin1_list: Sequence[str],
    keyword_filters: Sequence[str],
    keyword_match_all: bool,
    context_filters: Sequence[str],
    context_match_all: bool,
    lat_range: Sequence[float],
    lon_range: Sequence[float],
) -> pd.DataFrame:
    filtered = df.copy()
    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered[DATE_COL] >= pd.to_datetime(start)) & (filtered[DATE_COL] <= pd.to_datetime(end))]
    if countries:
        filtered = filtered[filtered[COUNTRY_COL].isin(countries)]
    if event_types:
        filtered = filtered[filtered[EVENT_TYPE_COL].isin(event_types)]
    if admin1_list:
        filtered = filtered[filtered[ADMIN1_COL].isin(admin1_list)]
    filtered = filtered[(filtered[LAT_COL].between(lat_range[0], lat_range[1])) & (filtered[LON_COL].between(lon_range[0], lon_range[1]))]
    filtered = filter_by_keywords(filtered, keyword_filters, [NOTES_COL, ACTOR1_COL, ADMIN1_COL, "location"], keyword_match_all)
    filtered = filter_by_context(filtered, context_filters, context_match_all)
    return filtered


def cluster_events(
    df: pd.DataFrame,
    features: Sequence[str],
    n_clusters: int,
) -> tuple[pd.DataFrame, float | None]:
    if df.empty or not features:
        return df.assign(cluster="Not computed"), None

    feature_df = df[features].copy()
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in features if col not in numeric_cols]

    transformed = []
    if numeric_cols:
        scaler = StandardScaler()
        numeric_vals = scaler.fit_transform(feature_df[numeric_cols])
        transformed.append(numeric_vals)

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(feature_df[categorical_cols])
        transformed.append(encoded)

    if not transformed:
        return df.assign(cluster="Not computed"), None

    matrix = np.hstack(transformed)
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    clusters = model.fit_predict(matrix)

    silhouette = None
    if n_clusters > 1 and len(df) > n_clusters:
        try:
            silhouette = float(silhouette_score(matrix, clusters))
        except Exception:
            silhouette = None

    clustered_df = df.copy()
    clustered_df["cluster"] = clusters.astype(str)
    return clustered_df, silhouette


def build_actor_network(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in df.iterrows():
        actor = row.get(ACTOR1_COL)
        assoc = row.get(ASSOC_ACTOR1_COL)
        if not actor or not assoc:
            continue
        actor = actor.strip()
        assoc = assoc.strip()
        if not actor or not assoc:
            continue
        if graph.has_edge(actor, assoc):
            graph[actor][assoc]["weight"] += 1
        else:
            graph.add_edge(actor, assoc, weight=1)
    return graph


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("ACLED Conflict Analytics Platform")
st.caption(
    "Explore geocoded events, discover patterns, and analyse conflict networks with the embedded ACLED dataset."
)

data = load_data()

with st.sidebar:
    st.header("Global filters")
    min_date, max_date = data[DATE_COL].min(), data[DATE_COL].max()
    date_range = st.date_input(
        "Event date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    countries = st.multiselect("Countries", options=sorted(data[COUNTRY_COL].dropna().unique()))
    admin1_options = sorted(data[ADMIN1_COL].dropna().unique())
    admin1_selection = st.multiselect("Admin 1 regions", options=admin1_options)
    event_types = st.multiselect("Event types", options=sorted(data[EVENT_TYPE_COL].dropna().unique()))

    st.markdown("---")
    st.subheader("Keyword search")
    keyword_raw = st.text_input("Keywords (comma separated)")
    keyword_logic = st.selectbox("Keyword match", ["Match any", "Match all"], index=0)

    st.subheader("Context in notes")
    context_raw = st.text_input("Context terms (comma separated)")
    context_logic = st.selectbox("Context match", ["Match all", "Match any"], index=0)

    st.markdown("---")
    lat_min, lat_max = float(data[LAT_COL].min()), float(data[LAT_COL].max())
    lon_min, lon_max = float(data[LON_COL].min()), float(data[LON_COL].max())
    lat_range = st.slider(
        "Latitude range",
        min_value=lat_min,
        max_value=lat_max,
        value=(lat_min, lat_max),
    )
    lon_range = st.slider(
        "Longitude range",
        min_value=lon_min,
        max_value=lon_max,
        value=(lon_min, lon_max),
    )

keyword_filters = parse_keywords(keyword_raw)
context_filters = parse_keywords(context_raw)

filtered = apply_filters(
    data,
    date_range,
    countries,
    event_types,
    admin1_selection,
    keyword_filters,
    keyword_logic == "Match all",
    context_filters,
    context_logic == "Match all",
    lat_range,
    lon_range,
)

st.success(f"Filtered dataset contains {len(filtered):,} events")

if filtered.empty:
    st.warning("No events match the selected filters. Adjust the parameters in the sidebar to continue.")
    st.stop()

overview_tab, search_tab, cluster_tab, network_tab, data_tab = st.tabs(
    ["Overview map", "Search insights", "Clustering", "Network analysis", "Data table"]
)

with overview_tab:
    st.subheader("Geocoded events")
    st.markdown("Visualise conflict events on an interactive map with optional aggregation by time period.")
    color_choice = st.selectbox("Colour events by", [EVENT_TYPE_COL, SUB_EVENT_COL, COUNTRY_COL, "year", "month"])
    size_choice = st.selectbox("Size events by", ["None", FATALITIES_COL])

    map_df = filtered.copy()
    if size_choice == "None":
        size_args = {}
    else:
        size_args = {"size": map_df[size_choice].clip(lower=0).fillna(0) + 5, "size_max": 20}

    fig = px.scatter_mapbox(
        map_df,
        lat=LAT_COL,
        lon=LON_COL,
        color=color_choice,
        hover_data={
            "event_id": map_df["event_id_cnty"],
            COUNTRY_COL: True,
            ADMIN1_COL: True,
            EVENT_TYPE_COL: True,
            SUB_EVENT_COL: True,
            FATALITIES_COL: True,
            NOTES_COL: True,
        },
        zoom=3,
        height=600,
        **size_args,
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Temporal trend")
    trend_freq = st.selectbox("Aggregate by", ["week", "month", "year"], index=1)
    trend_df = (
        filtered.groupby(trend_freq)
        .agg(events=("event_id_cnty", "count"), fatalities=(FATALITIES_COL, "sum"))
        .reset_index()
        .sort_values(trend_freq)
    )
    line_fig = px.line(trend_df, x=trend_freq, y="events", markers=True)
    line_fig.update_layout(yaxis_title="Number of events", xaxis_title=trend_freq.title())
    st.plotly_chart(line_fig, use_container_width=True)

with search_tab:
    st.subheader("Keyword search results")
    if keyword_filters:
        st.write("Matching events for keywords:")
        st.code(", ".join(keyword_filters), language="text")
    else:
        st.info("Add keywords from the sidebar to search across notes, actors, and locations.")

    search_cols = ["event_id_cnty", DATE_COL, COUNTRY_COL, ADMIN1_COL, EVENT_TYPE_COL, SUB_EVENT_COL, ACTOR1_COL, NOTES_COL]
    st.dataframe(filtered[search_cols].head(200), use_container_width=True, hide_index=True)

    st.subheader("Contextual highlights")
    if context_filters:
        context_counts = (
            filtered.assign(match_context=lambda x: x[NOTES_COL].str.lower())
            .assign(matches=lambda x: x["match_context"].apply(lambda txt: [ctx for ctx in context_filters if ctx.lower() in txt]))
        )
        context_summary = (
            context_counts.explode("matches")
            .dropna(subset=["matches"])
            .groupby("matches")
            .agg(events=("event_id_cnty", "count"), fatalities=(FATALITIES_COL, "sum"))
            .reset_index()
        )
        if context_summary.empty:
            st.warning("No contextual matches found in the filtered events.")
        else:
            st.table(context_summary.sort_values("events", ascending=False))
    else:
        st.info("Add context terms from the sidebar to analyse themes in the notes column.")

with cluster_tab:
    st.subheader("Cluster events by attributes")
    st.markdown(
        "Group events using K-means clustering across spatial, temporal, and categorical features."
    )
    available_features = [LAT_COL, LON_COL, FATALITIES_COL, "year", EVENT_TYPE_COL, SUB_EVENT_COL, ADMIN1_COL]
    selected_features = st.multiselect(
        "Select features for clustering",
        options=available_features,
        default=[LAT_COL, LON_COL, FATALITIES_COL],
    )
    cluster_count = st.slider("Number of clusters", min_value=2, max_value=10, value=4)
    run_cluster = st.button("Run clustering")

    if run_cluster:
        clustered_df, silhouette = cluster_events(filtered, selected_features, cluster_count)
        if "cluster" not in clustered_df.columns or clustered_df["cluster"].isna().all():
            st.warning("Unable to compute clusters with the current selection.")
        else:
            st.success("Clustering complete.")
            if silhouette is not None:
                st.metric("Silhouette score", f"{silhouette:.3f}")
            cluster_counts = clustered_df.groupby("cluster").agg(
                events=("event_id_cnty", "count"),
                mean_fatalities=(FATALITIES_COL, "mean"),
            )
            st.dataframe(cluster_counts, use_container_width=True)

            cluster_fig = px.scatter_mapbox(
                clustered_df,
                lat=LAT_COL,
                lon=LON_COL,
                color="cluster",
                hover_name="event_id_cnty",
                hover_data={EVENT_TYPE_COL: True, SUB_EVENT_COL: True, FATALITIES_COL: True, NOTES_COL: True},
                zoom=3,
                height=600,
            )
            cluster_fig.update_layout(mapbox_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
            st.plotly_chart(cluster_fig, use_container_width=True)
    else:
        st.info("Select features and click 'Run clustering' to generate event clusters.")

with network_tab:
    st.subheader("Actor association network")
    st.markdown(
        "Construct a co-occurrence network from primary actors and their associated actors within events."
    )
    graph = build_actor_network(filtered)
    if graph.number_of_edges() == 0:
        st.warning("Not enough actor association data to build a network for the selected filters.")
    else:
        top_n = st.slider("Show top actors", min_value=5, max_value=30, value=10)
        degree_series = pd.Series(dict(graph.degree(weight="weight")))
        top_nodes = degree_series.sort_values(ascending=False).head(top_n).index.tolist()
        subgraph = graph.subgraph(top_nodes)
        pos = nx.spring_layout(subgraph, weight="weight", seed=42)

        edge_x, edge_y = [], []
        for src, dst in subgraph.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        node_size = [subgraph.degree(node, weight="weight") * 10 for node in subgraph.nodes()]

        network_fig = px.scatter(
            x=node_x,
            y=node_y,
            text=list(subgraph.nodes()),
            size=node_size,
        )
        network_fig.update_traces(marker=dict(color="#3182bd", line=dict(width=1, color="#ffffff")))
        network_fig.add_scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#9ecae1", width=1), hoverinfo="none")
        network_fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(network_fig, use_container_width=True)

        centrality = nx.degree_centrality(subgraph)
        centrality_df = (
            pd.DataFrame({"actor": list(centrality.keys()), "centrality": list(centrality.values())})
            .sort_values("centrality", ascending=False)
        )
        st.table(centrality_df)

with data_tab:
    st.subheader("Filtered dataset")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="acled_filtered_events.csv",
        mime="text/csv",
    )


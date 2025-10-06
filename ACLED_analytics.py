# -*- coding: utf-8 -*-
"""Streamlit analytic tool for ACLED data.

Created on Mon Oct  6 15:38:27 2025
Author: LWu
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import textwrap

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


@st.cache_resource(show_spinner=False)
def build_semantic_index(notes: pd.Series):
    """Build a TF-IDF index to enable semantic matching on notes."""
    prepared_notes = notes.fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(prepared_notes)
    index_lookup = {idx: position for position, idx in enumerate(prepared_notes.index)}
    return vectorizer, matrix, index_lookup


@st.cache_resource(show_spinner=False)
def build_context_matrix(notes: pd.Series):
    """Create a compact TF-IDF representation of notes for contextual clustering."""
    prepared_notes = notes.fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=300, min_df=5)
    matrix = vectorizer.fit_transform(prepared_notes)
    index_lookup = {idx: position for position, idx in enumerate(prepared_notes.index)}
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, matrix, index_lookup, feature_names


def parse_keywords(raw: str) -> list[str]:
    return [kw.strip() for kw in (raw or "").split(",") if kw.strip()]


def filter_by_keywords(
    df: pd.DataFrame,
    keywords: Sequence[str],
    columns: Sequence[str],
    match_all: bool = False,
) -> pd.DataFrame:
    if not keywords:
        return df

    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        return df

    mask = pd.Series(False, index=df.index) if not match_all else pd.Series(True, index=df.index)
    for kw in keywords:
        col_mask = pd.Series(False, index=df.index)
        for col in available_columns:
            series = df[col].fillna("").astype(str)
            col_mask |= series.str.contains(kw, case=False, na=False)
        mask = mask & col_mask if match_all else mask | col_mask

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


def semantic_search(
    df: pd.DataFrame,
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    index_lookup: dict,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the most semantically similar events for a user query."""
    if not query.strip():
        return pd.DataFrame(columns=df.columns.tolist() + ["semantic_score"])

    query_vec = vectorizer.transform([query])
    subset_positions: list[tuple[int, int]] = []
    for df_position, idx in enumerate(df.index):
        matrix_position = index_lookup.get(idx)
        if matrix_position is not None:
            subset_positions.append((df_position, matrix_position))

    if not subset_positions:
        return pd.DataFrame(columns=df.columns.tolist() + ["semantic_score"])

    df_positions, matrix_positions = map(np.array, zip(*subset_positions))
    subset_matrix = matrix[matrix_positions]
    similarity = cosine_similarity(subset_matrix, query_vec).ravel()

    if np.allclose(similarity, 0):
        return pd.DataFrame(columns=df.columns.tolist() + ["semantic_score"])

    order = np.argsort(similarity)[::-1][:top_k]
    ranked_df = df.iloc[df_positions[order]].copy()
    ranked_df["semantic_score"] = similarity[order]
    return ranked_df


def contextual_feature_matrix(
    df: pd.DataFrame,
    context_matrix,
    index_lookup: dict,
) -> np.ndarray | None:
    """Extract contextual embeddings for the provided dataframe."""
    subset_positions: list[int] = []
    for idx in df.index:
        matrix_position = index_lookup.get(idx)
        if matrix_position is not None:
            subset_positions.append(matrix_position)
    if not subset_positions:
        return None
    subset = context_matrix[subset_positions]
    return subset.toarray()


def craft_event_story(row: pd.Series) -> str:
    """Return a compact narrative used for map and network tooltips."""
    date_value = row.get(DATE_COL)
    date_text = "Unknown date"
    if pd.notnull(date_value):
        date_text = pd.to_datetime(date_value).strftime("%d %b %Y")

    location_bits = [str(val) for val in [row.get(ADMIN1_COL), row.get(COUNTRY_COL)] if val]
    location_text = ", ".join(location_bits) if location_bits else "Location unavailable"

    fatalities = row.get(FATALITIES_COL)
    if pd.notnull(fatalities):
        fatal_int = int(fatalities)
        fatal_text = f"{fatal_int} fatality" if fatal_int == 1 else f"{fatal_int} fatalities"
    else:
        fatal_text = "Fatalities not reported"

    notes = textwrap.shorten(str(row.get(NOTES_COL, "")).strip(), width=180, placeholder="…")
    actor = row.get(ACTOR1_COL) or "Unknown actor"
    event_text = row.get(EVENT_TYPE_COL) or "Event"
    sub_event = row.get(SUB_EVENT_COL) or ""
    sub_event_text = f" ({sub_event})" if sub_event else ""

    story = (
        f"<b>{date_text}</b><br>"
        f"<span style='color:#4B5563'>{location_text}</span><br>"
        f"<b>Primary actor:</b> {actor}<br>"
        f"<b>Event:</b> {event_text}{sub_event_text}<br>"
        f"<b>Impact:</b> {fatal_text}<br>"
        f"<b>Context:</b> {notes}"
    )
    return story


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


def cluster_events(
    df: pd.DataFrame,
    features: Sequence[str],
    n_clusters: int,
    contextual_features: np.ndarray | None = None,
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

    if contextual_features is not None:
        transformed.append(contextual_features)

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


# ---------------------------------------------------------------------------
# Sidebar selections
# ---------------------------------------------------------------------------
@dataclass
class FilterState:
    date_range: Sequence[pd.Timestamp]
    countries: list[str]
    admin1_values: list[str]
    event_types: list[str]
    keyword_filters: list[str]
    keyword_match_all: bool
    context_filters: list[str]
    context_match_all: bool
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]


def render_sidebar(data: pd.DataFrame) -> FilterState:
    with st.sidebar:
        st.header("Global filters")
        min_date_raw = pd.to_datetime(data[DATE_COL].min())
        max_date_raw = pd.to_datetime(data[DATE_COL].max())
        min_date = min_date_raw.date() if pd.notnull(min_date_raw) else None
        max_date = max_date_raw.date() if pd.notnull(max_date_raw) else None
        if not (min_date and max_date):
            today = pd.Timestamp.utcnow().date()
            min_date = max_date = today
        default_range = (min_date, max_date)
        date_range = st.date_input(
            "Event date range",
            value=default_range,
            min_value=min_date,
            max_value=max_date,
        )

        countries = st.multiselect("Countries", options=sorted(data[COUNTRY_COL].dropna().unique()))
        if countries:
            admin_subset = (
                data[data[COUNTRY_COL].isin(countries)][[COUNTRY_COL, ADMIN1_COL]]
                .dropna()
                .drop_duplicates()
                .sort_values([COUNTRY_COL, ADMIN1_COL])
            )
        else:
            admin_subset = (
                data[[COUNTRY_COL, ADMIN1_COL]]
                .dropna()
                .drop_duplicates()
                .sort_values([COUNTRY_COL, ADMIN1_COL])
            )
        admin1_options = list(admin_subset.itertuples(index=False, name=None))
        admin1_selection = st.multiselect(
            "Admin 1 regions",
            options=admin1_options,
            format_func=lambda item: f"{item[0]} — {item[1]}",
            help="Selections reflect the chosen countries; deselect countries to browse all Admin 1 regions.",
        )
        admin1_values = [item[1] for item in admin1_selection]

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

    return FilterState(
        date_range=date_range,
        countries=list(countries),
        admin1_values=admin1_values,
        event_types=list(event_types),
        keyword_filters=parse_keywords(keyword_raw),
        keyword_match_all=keyword_logic == "Match all",
        context_filters=parse_keywords(context_raw),
        context_match_all=context_logic == "Match all",
        lat_range=(float(lat_range[0]), float(lat_range[1])),
        lon_range=(float(lon_range[0]), float(lon_range[1])),
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    filtered = df.copy()
    if state.date_range and len(state.date_range) == 2:
        start, end = state.date_range
        filtered = filtered[(filtered[DATE_COL] >= pd.to_datetime(start)) & (filtered[DATE_COL] <= pd.to_datetime(end))]
    if state.countries:
        filtered = filtered[filtered[COUNTRY_COL].isin(state.countries)]
    if state.event_types:
        filtered = filtered[filtered[EVENT_TYPE_COL].isin(state.event_types)]
    if state.admin1_values:
        filtered = filtered[filtered[ADMIN1_COL].isin(state.admin1_values)]

    filtered = filtered[
        (filtered[LAT_COL].between(state.lat_range[0], state.lat_range[1]))
        & (filtered[LON_COL].between(state.lon_range[0], state.lon_range[1]))
    ]
    filtered = filter_by_keywords(filtered, state.keyword_filters, [NOTES_COL, ACTOR1_COL, ADMIN1_COL, "location"], state.keyword_match_all)
    filtered = filter_by_context(filtered, state.context_filters, state.context_match_all)
    return filtered


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_landing_tab():
    st.subheader("Getting started")
    st.markdown(
        """
        Welcome to the ACLED Conflict Analytics Platform. The application preloads the curated
        ACLED dataset bundled with this tool, so you can begin exploring immediately.

        **How to navigate the app**

        1. Use the **global filters** in the sidebar to focus on specific timelines, countries,
           administrative regions, or thematic keywords. Spatial sliders help you spotlight
           latitude and longitude ranges without leaving the page.
        2. The **Overview map** blends an interactive map with temporal trendlines so you can
           see where and when events occur.
        3. **Search insights** combines traditional keyword filters with semantic search powered by
           natural-language processing to surface relevant narratives.
        4. Discover clusters of similar events—including optional context-aware groupings—in the
           **Clustering** tab.
        5. Reveal actor relationships and central players in the **Network analysis** view.
        6. Export your working dataset at any point from the **Data table** tab.

        Tip: hover over map markers or network nodes to read concise stories crafted for each event.
        """
    )


def render_overview_tab(filtered: pd.DataFrame):
    st.subheader("Geocoded events")
    st.markdown("Visualise conflict events on an interactive map with optional aggregation by time period.")
    color_choice = st.selectbox("Colour events by", [EVENT_TYPE_COL, SUB_EVENT_COL, COUNTRY_COL, "year", "month"])
    size_choice = st.selectbox("Size events by", ["None", FATALITIES_COL])

    map_df = filtered.copy()
    map_df["event_story"] = map_df.apply(craft_event_story, axis=1)
    size_args: dict[str, float] = {}
    if size_choice != "None":
        size_args = {"size": map_df[size_choice].clip(lower=0).fillna(0) + 5, "size_max": 20}

    fig = px.scatter_mapbox(
        map_df,
        lat=LAT_COL,
        lon=LON_COL,
        color=color_choice,
        hover_data=None,
        custom_data=["event_story"],
        zoom=3,
        height=600,
        **size_args,
    )
    marker_style = dict(opacity=0.82)
    if size_choice == "None":
        marker_style["size"] = 10
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker=marker_style)
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend=dict(orientation="h", yanchor="bottom", y=0.99, x=0, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})

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


def render_search_tab(
    filtered: pd.DataFrame,
    filter_state: FilterState,
    semantic_vectorizer: TfidfVectorizer,
    semantic_matrix,
    semantic_index: dict,
    context_vectorizer: TfidfVectorizer,
    context_matrix,
    context_index: dict,
):
    st.subheader("Keyword search results")
    if filter_state.keyword_filters:
        st.write("Matching events for keywords:")
        st.code(", ".join(filter_state.keyword_filters), language="text")
    else:
        st.info("Add keywords from the sidebar to search across notes, actors, and locations.")

    search_cols = [
        "event_id_cnty",
        DATE_COL,
        COUNTRY_COL,
        ADMIN1_COL,
        EVENT_TYPE_COL,
        SUB_EVENT_COL,
        ACTOR1_COL,
        NOTES_COL,
    ]
    st.dataframe(filtered[search_cols].head(200), use_container_width=True, hide_index=True)

    st.subheader("Semantic search (NLP)")
    st.markdown(
        "Describe the type of incident you are investigating to retrieve semantically similar events, even when exact keywords differ."
    )
    semantic_query = st.text_input("Semantic query", placeholder="e.g. attacks on aid workers near border crossings")
    semantic_limit = st.slider("Number of semantic matches", min_value=5, max_value=50, value=15, step=5)
    if semantic_query:
        semantic_results = semantic_search(
            filtered,
            semantic_query,
            semantic_vectorizer,
            semantic_matrix,
            semantic_index,
            top_k=semantic_limit,
        )
        if semantic_results.empty:
            st.warning("No semantic matches were found for this query within the filtered events.")
        else:
            semantic_results = semantic_results.assign(event_story=lambda x: x.apply(craft_event_story, axis=1))
            top_three = semantic_results.head(3)
            for _, row in top_three.iterrows():
                st.markdown(f"{row['event_story']}", unsafe_allow_html=True)
                st.caption(f"Semantic similarity score: {row['semantic_score']:.3f}")
            st.dataframe(
                semantic_results[
                    [
                        "semantic_score",
                        DATE_COL,
                        COUNTRY_COL,
                        ADMIN1_COL,
                        EVENT_TYPE_COL,
                        SUB_EVENT_COL,
                        ACTOR1_COL,
                        NOTES_COL,
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Enter a semantic query above to activate contextual matching.")

    st.subheader("Contextual highlights")
    context_positions = [context_index.get(idx) for idx in filtered.index if context_index.get(idx) is not None]
    if context_positions:
        term_strength = np.asarray(context_matrix[context_positions].sum(axis=0)).ravel()
        top_context_ids = term_strength.argsort()[::-1][:10]
        top_context_terms = pd.DataFrame(
            {
                "context": context_vectorizer.get_feature_names_out()[top_context_ids],
                "relevance": term_strength[top_context_ids],
            }
        )
        st.table(top_context_terms)
        st.caption("Top contextual themes by TF-IDF weight within the filtered events.")
    else:
        st.info("Contextual term summaries will appear once events are available in the filtered set.")

    if filter_state.context_filters:
        context_counts = (
            filtered.assign(match_context=lambda x: x[NOTES_COL].str.lower())
            .assign(matches=lambda x: x["match_context"].apply(lambda txt: [ctx for ctx in filter_state.context_filters if ctx.lower() in txt]))
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


def render_clustering_tab(
    filtered: pd.DataFrame,
    context_matrix,
    context_index: dict,
):
    st.subheader("Cluster events by attributes")
    st.markdown("Group events using K-means clustering across spatial, temporal, and categorical features.")
    available_features = [LAT_COL, LON_COL, FATALITIES_COL, "year", EVENT_TYPE_COL, SUB_EVENT_COL, ADMIN1_COL]
    selected_features = st.multiselect(
        "Select features for clustering",
        options=available_features,
        default=[LAT_COL, LON_COL, FATALITIES_COL],
    )
    use_context_topics = st.checkbox(
        "Incorporate context from notes",
        value=False,
        help="Augment clustering with TF-IDF embeddings derived from the notes column to group events with similar narratives.",
    )
    cluster_count = st.slider("Number of clusters", min_value=2, max_value=10, value=4)
    run_cluster = st.button("Run clustering")

    if not run_cluster:
        st.info("Select features and click 'Run clustering' to generate event clusters.")
        return

    contextual_features = None
    if use_context_topics:
        contextual_features = contextual_feature_matrix(filtered, context_matrix, context_index)
        if contextual_features is None:
            st.warning("Contextual embeddings could not be generated for the current selection.")

    clustered_df, silhouette = cluster_events(
        filtered,
        selected_features,
        cluster_count,
        contextual_features=contextual_features,
    )
    if "cluster" not in clustered_df.columns or clustered_df["cluster"].isna().all():
        st.warning("Unable to compute clusters with the current selection.")
        return

    st.success("Clustering complete.")
    if silhouette is not None:
        st.metric("Silhouette score", f"{silhouette:.3f}")
    cluster_counts = clustered_df.groupby("cluster").agg(
        events=("event_id_cnty", "count"),
        mean_fatalities=(FATALITIES_COL, "mean"),
    )
    st.dataframe(cluster_counts, use_container_width=True)

    cluster_display = clustered_df.assign(event_story=lambda x: x.apply(craft_event_story, axis=1))
    cluster_fig = px.scatter_mapbox(
        cluster_display,
        lat=LAT_COL,
        lon=LON_COL,
        color="cluster",
        hover_name=None,
        hover_data=None,
        custom_data=["event_story", "cluster"],
        zoom=3,
        height=600,
    )
    cluster_fig.update_traces(
        hovertemplate="Cluster %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
        marker=dict(opacity=0.8, size=10),
    )
    cluster_fig.update_layout(mapbox_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(cluster_fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})


def render_network_tab(filtered: pd.DataFrame):
    st.subheader("Actor association network")
    st.markdown("Construct a co-occurrence network from primary actors and their associated actors within events.")
    graph = build_actor_network(filtered)
    if graph.number_of_edges() == 0:
        st.warning("Not enough actor association data to build a network for the selected filters.")
        return

    top_n = st.slider("Show top actors", min_value=5, max_value=30, value=10)
    degree_series = pd.Series(dict(graph.degree(weight="weight")))
    top_nodes = degree_series.sort_values(ascending=False).head(top_n).index.tolist()
    subgraph = graph.subgraph(top_nodes)
    centrality = nx.degree_centrality(subgraph)
    weighted_degree = dict(subgraph.degree(weight="weight"))
    pos = nx.spring_layout(subgraph, weight="weight", seed=42)

    if centrality:
        max_centrality = max(centrality.values()) or 1
        for node, coords in pos.items():
            scale = 0.35 + (1 - (centrality[node] / max_centrality))
            pos[node] = np.array(coords) * scale

    edge_traces: list[go.Scatter] = []
    for src, dst, attrs in subgraph.edges(data=True):
        weight = attrs.get("weight", 1)
        edge_trace = go.Scatter(
            x=[pos[src][0], pos[dst][0]],
            y=[pos[src][1], pos[dst][1]],
            mode="lines",
            line=dict(width=1 + weight * 0.2, color="#c6dbef"),
            hoverinfo="text",
            text=f"{src} ↔ {dst}<br>Interactions: {weight}",
        )
        edge_traces.append(edge_trace)

    node_trace = go.Scatter(
        x=[pos[node][0] for node in subgraph.nodes()],
        y=[pos[node][1] for node in subgraph.nodes()],
        mode="markers+text",
        text=[node for node in subgraph.nodes()],
        textposition="top center",
        marker=dict(
            size=[12 + centrality.get(node, 0) * 80 for node in subgraph.nodes()],
            color="#3182bd",
            line=dict(width=1.5, color="#f7fbff"),
        ),
        hovertemplate=[
            f"<b>{node}</b><br>Weighted degree: {weighted_degree.get(node, 0):.0f}<br>Centrality: {centrality.get(node, 0):.3f}<extra></extra>"
            for node in subgraph.nodes()
        ],
    )

    network_fig = go.Figure(data=edge_traces + [node_trace])
    network_fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(network_fig, use_container_width=True, config={"displayModeBar": False})

    centrality_df = (
        pd.DataFrame(
            {
                "actor": list(subgraph.nodes()),
                "weighted_degree": [weighted_degree.get(node, 0) for node in subgraph.nodes()],
                "centrality": [centrality.get(node, 0) for node in subgraph.nodes()],
            }
        )
        .sort_values("centrality", ascending=False)
        .reset_index(drop=True)
    )
    st.table(centrality_df)


def render_data_tab(filtered: pd.DataFrame):
    st.subheader("Filtered dataset")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="acled_filtered_events.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    st.title("ACLED Conflict Analytics Platform")
    st.caption("Explore geocoded events, discover patterns, and analyse conflict networks with the embedded ACLED dataset.")

    data = load_data()
    semantic_vectorizer, semantic_matrix, semantic_index = build_semantic_index(data[NOTES_COL])
    context_vectorizer, context_matrix, context_index, _ = build_context_matrix(data[NOTES_COL])

    filter_state = render_sidebar(data)
    filtered = apply_filters(data, filter_state)

    st.success(f"Filtered dataset contains {len(filtered):,} events")

    if filtered.empty:
        st.warning("No events match the selected filters. Adjust the parameters in the sidebar to continue.")
        st.stop()

    guide_tab, overview_tab, search_tab, cluster_tab, network_tab, data_tab = st.tabs(
        [
            "Guide",
            "Overview map",
            "Search insights",
            "Clustering",
            "Network analysis",
            "Data table",
        ]
    )

    with guide_tab:
        render_landing_tab()
    with overview_tab:
        render_overview_tab(filtered)
    with search_tab:
        render_search_tab(
            filtered,
            filter_state,
            semantic_vectorizer,
            semantic_matrix,
            semantic_index,
            context_vectorizer,
            context_matrix,
            context_index,
        )
    with cluster_tab:
        render_clustering_tab(filtered, context_matrix, context_index)
    with network_tab:
        render_network_tab(filtered)
    with data_tab:
        render_data_tab(filtered)


if __name__ == "__main__":
    main()

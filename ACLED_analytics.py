# -*- coding: utf-8 -*-
"""Streamlit analytic tool for ACLED data.

Created on Mon Oct  6 15:38:27 2025
Author: LWu
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import textwrap

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency for language-guided clustering
    SentenceTransformer = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "ACLED 2016-2025.xlsx"
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


# Colour palettes
QUALITATIVE_SCHEMES = {
    EVENT_TYPE_COL: px.colors.qualitative.Set2,
    SUB_EVENT_COL: px.colors.qualitative.Safe,
    COUNTRY_COL: px.colors.qualitative.Dark24,
    "year": px.colors.qualitative.Prism,
    "month": px.colors.qualitative.Prism,
    "cluster": px.colors.qualitative.Bold,
}


MONTH_TERMS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


GENERIC_CONTEXT_STOPWORDS = {
    "according",
    "alleged",
    "authorities",
    "community",
    "district",
    "local",
    "reported",
    "residents",
    "security",
    "said",
    "state",
    "villagers",
}


# ---------------------------------------------------------------------------
# Data loading & utilities
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load ACLED data from disk with typed columns."""
    df = pd.read_excel(DATA_PATH)
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


def is_conflict_term(term: str) -> bool:
    """Return True when a TF-IDF term is conflict-specific rather than temporal."""
    normalised = term.lower().replace("_", " ")
    parts = [token for token in re.split(r"[\s\-/]+", normalised) if token]
    if not parts:
        return False
    for token in parts:
        if any(char.isdigit() for char in token):
            return False
        if token in MONTH_TERMS:
            return False
        if len(token) <= 2:
            return False
    if all(token in GENERIC_CONTEXT_STOPWORDS for token in parts):
        return False
    return True


def select_top_term_indices(term_strength: np.ndarray, feature_names: np.ndarray, limit: int) -> list[int]:
    """Return indices of the strongest conflict-relevant TF-IDF terms."""
    sorted_indices = term_strength.argsort()[::-1]
    selected: list[int] = []
    for idx in sorted_indices:
        if term_strength[idx] <= 0:
            continue
        term = feature_names[idx]
        if not is_conflict_term(term):
            continue
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def build_note_embeddings(notes: pd.Series):
    model = load_sentence_model()
    if model is None:
        return None, None
    prepared_notes = notes.fillna("").astype(str)
    try:
        embeddings = model.encode(
            prepared_notes.tolist(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except TypeError:
        embeddings = model.encode(
            prepared_notes.tolist(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Normalise manually if the library version does not support the parameter.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
    index_lookup = {idx: position for position, idx in enumerate(prepared_notes.index)}
    return embeddings, index_lookup


def align_embeddings(df: pd.DataFrame, embeddings, index_lookup: dict):
    if embeddings is None or index_lookup is None:
        return None, None
    subset_positions: list[tuple[int, int]] = []
    for df_position, idx in enumerate(df.index):
        matrix_position = index_lookup.get(idx)
        if matrix_position is not None:
            subset_positions.append((df_position, matrix_position))
    if not subset_positions:
        return None, None
    df_positions, matrix_positions = map(np.array, zip(*subset_positions))
    subset_embeddings = embeddings[matrix_positions]
    return subset_embeddings, df_positions


def language_guided_clustering(
    df: pd.DataFrame,
    note_embeddings,
    note_index_lookup: dict,
    query: str,
    cluster_count: int,
    max_events: int = 250,
):
    """Cluster events guided by a natural-language query using sentence-transformer embeddings."""
    model = load_sentence_model()
    if model is None:
        return None, "model_unavailable"
    query = query.strip()
    if not query:
        return None, "empty_query"
    subset_embeddings, df_positions = align_embeddings(df, note_embeddings, note_index_lookup)
    if subset_embeddings is None:
        return None, "no_embeddings"
    try:
        query_embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
    except TypeError:
        query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
    similarities = subset_embeddings @ query_embedding
    if not np.any(similarities > 0):
        return None, "no_similarity"

    order = np.argsort(similarities)[::-1][:max_events]
    ordered_scores = similarities[order]
    positive_mask = ordered_scores > 0.05
    selected_indices = order[positive_mask]
    if selected_indices.size < cluster_count * 2:
        broader_mask = ordered_scores > 0
        selected_indices = order[broader_mask]
    if selected_indices.size < cluster_count * 2:
        return None, "insufficient_events"

    selected_embeddings = subset_embeddings[selected_indices]
    selected_positions = df_positions[selected_indices]
    selected_df = df.iloc[selected_positions].copy()
    selected_df["similarity"] = similarities[selected_indices]

    adjusted_clusters = min(cluster_count, max(2, selected_df.shape[0] // 3))
    if adjusted_clusters < 2 or selected_df.shape[0] < adjusted_clusters:
        return None, "insufficient_events"

    try:
        model_kmeans = KMeans(n_clusters=adjusted_clusters, random_state=42, n_init="auto")
    except TypeError:
        model_kmeans = KMeans(n_clusters=adjusted_clusters, random_state=42, n_init=10)
    labels = model_kmeans.fit_predict(selected_embeddings)
    selected_df["cluster"] = labels.astype(str)
    return selected_df, "ok"


def build_colour_mapping(df: pd.DataFrame, column: str, palette_key: str | None = None) -> dict:
    """Return a consistent colour mapping for a categorical column."""
    key = palette_key or column
    palette = QUALITATIVE_SCHEMES.get(key)
    if palette is None or column not in df.columns:
        return {}
    unique_values = [value for value in df[column].dropna().astype(str).sort_values().unique()]
    colour_cycle = itertools.cycle(palette)
    return {value: next(colour_cycle) for value in unique_values}


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

    # Apply conservative filtering to reduce semantic false positives.
    filtered_ranked_df = apply_semantic_filters(ranked_df, query)
    similarity_threshold = 0.12
    filtered_ranked_df = filtered_ranked_df[filtered_ranked_df["semantic_score"] >= similarity_threshold]
    return filtered_ranked_df


SEMANTIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "onto",
    "over",
    "under",
    "about",
    "after",
    "before",
    "between",
    "without",
    "against",
    "towards",
    "toward",
    "among",
    "into",
    "per",
    "within",
}


def apply_semantic_filters(results: pd.DataFrame, query: str) -> pd.DataFrame:
    """Reduce semantic matches that do not contain the key query concepts."""

    if results.empty:
        return results

    query = query.lower()
    raw_tokens = re.findall(r"[\w']+", query)
    tokens = [token for token in raw_tokens if len(token) > 2 and token not in SEMANTIC_STOPWORDS]
    if not tokens:
        return results

    # Expand tokens to include simple morphological variants for stricter containment checks.
    expanded_tokens: set[str] = set()
    for token in tokens:
        expanded_tokens.add(token)
        if token.endswith("es") and len(token) > 3:
            expanded_tokens.add(token[:-2])
        if token.endswith("s") and len(token) > 3:
            expanded_tokens.add(token[:-1])
        if token.endswith("ed") and len(token) > 3:
            expanded_tokens.add(token[:-2])
        if token.endswith("ing") and len(token) > 4:
            expanded_tokens.add(token[:-3])
    expanded_tokens_list = sorted(expanded_tokens)

    phrases = [
        f"{raw_tokens[i]} {raw_tokens[i + 1]}"
        for i in range(len(raw_tokens) - 1)
        if raw_tokens[i] not in SEMANTIC_STOPWORDS or raw_tokens[i + 1] not in SEMANTIC_STOPWORDS
    ]

    required_token_matches = len(expanded_tokens_list) if len(expanded_tokens_list) > 1 else 1

    def note_matches(note: str) -> bool:
        lowered = note.lower()
        if any(phrase in lowered for phrase in phrases):
            return True
        matches = sum(token in lowered for token in expanded_tokens_list)
        return matches >= required_token_matches

    mask = results[NOTES_COL].fillna("").astype(str).apply(note_matches)
    return results[mask]


def align_context_matrix(
    df: pd.DataFrame,
    context_matrix,
    context_index: dict,
):
    """Align the TF-IDF matrix with dataframe rows."""
    df_positions: list[int] = []
    matrix_positions: list[int] = []
    for df_position, idx in enumerate(df.index):
        matrix_position = context_index.get(idx)
        if matrix_position is None:
            continue
        df_positions.append(df_position)
        matrix_positions.append(matrix_position)
    if not matrix_positions:
        return None, None
    subset = context_matrix[matrix_positions]
    return subset, np.array(df_positions)


def contextual_feature_matrix(
    df: pd.DataFrame,
    context_matrix,
    index_lookup: dict,
    n_components: int = 12,
) -> tuple[np.ndarray | None, TruncatedSVD | None]:
    """Extract contextual embeddings via truncated SVD for the provided dataframe."""
    subset, df_positions = align_context_matrix(df, context_matrix, index_lookup)
    if subset is None:
        return None, None

    max_components = min(n_components, subset.shape[0] - 1, subset.shape[1] - 1)
    if max_components <= 0:
        return None, None

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    reduced = svd.fit_transform(subset)

    contextual_features = np.zeros((len(df), reduced.shape[1]))
    contextual_features[df_positions] = reduced
    return contextual_features, svd


def build_context_summary(
    df: pd.DataFrame,
    context_matrix,
    context_index: dict,
    context_vectorizer: TfidfVectorizer,
    top_terms: int = 8,
) -> pd.DataFrame:
    """Create a contextual summary table with representative events for top TF-IDF themes."""
    subset, df_positions = align_context_matrix(df, context_matrix, context_index)
    if subset is None:
        return pd.DataFrame()

    term_strength = np.asarray(subset.sum(axis=0)).ravel()
    if not np.any(term_strength):
        return pd.DataFrame()

    feature_names = context_vectorizer.get_feature_names_out()
    top_indices = select_top_term_indices(term_strength, feature_names, top_terms)
    if not top_indices:
        return pd.DataFrame()

    rows: list[dict] = []
    for term_idx in top_indices:
        weight = float(term_strength[term_idx])
        if weight <= 0:
            continue
        column = subset[:, term_idx].toarray().ravel()
        if not np.any(column):
            continue
        event_position = int(column.argmax())
        df_position = int(df_positions[event_position])
        event_row = df.iloc[df_position]
        date_value = event_row.get(DATE_COL)
        date_text = "Unknown"
        if pd.notnull(date_value):
            date_text = pd.to_datetime(date_value).strftime("%Y-%m-%d")
        location_bits = [event_row.get(ADMIN1_COL), event_row.get(COUNTRY_COL)]
        location = ", ".join([str(bit) for bit in location_bits if bit]) or "Location unavailable"
        rows.append(
            {
                "theme": feature_names[term_idx],
                "salience": round(weight, 3),
                "event_date": date_text,
                "location": location,
                "representative_note": textwrap.shorten(
                    str(event_row.get(NOTES_COL, "")).strip(), width=160, placeholder="…"
                ),
            }
        )

    return pd.DataFrame(rows)


def summarise_cluster_contexts(
    clustered_df: pd.DataFrame,
    context_matrix,
    context_index: dict,
    context_vectorizer: TfidfVectorizer,
    top_terms: int = 5,
) -> pd.DataFrame:
    """Summarise clusters with dominant note themes."""
    if "cluster" not in clustered_df.columns:
        return pd.DataFrame()

    summaries: list[dict] = []
    for cluster_label, group in clustered_df.groupby("cluster"):
        subset, _ = align_context_matrix(group, context_matrix, context_index)
        if subset is None:
            continue
        term_strength = np.asarray(subset.sum(axis=0)).ravel()
        if not np.any(term_strength):
            continue
        feature_names = context_vectorizer.get_feature_names_out()
        top_indices = select_top_term_indices(term_strength, feature_names, top_terms)
        top_terms_list = [feature_names[idx] for idx in top_indices]
        summaries.append(
            {
                "cluster": str(cluster_label),
                "events": len(group),
                "dominant_themes": ", ".join(top_terms_list),
            }
        )

    return pd.DataFrame(summaries)


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
        actor = str(row.get(ACTOR1_COL) or "").strip()
        assoc = str(row.get(ASSOC_ACTOR1_COL) or "").strip()
        if not actor or not assoc:
            continue
        fatalities_value = row.get(FATALITIES_COL)
        fatality = float(fatalities_value) if pd.notnull(fatalities_value) else 0.0
        for node in (actor, assoc):
            if not graph.has_node(node):
                graph.add_node(node, fatalities=0.0, events=0)
            graph.nodes[node]["fatalities"] += fatality
            graph.nodes[node]["events"] += 1
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

    if len(df) < max(n_clusters, 2):
        return df.assign(cluster="Not computed"), None

    feature_df = df[list(features)].copy()
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in features if col not in numeric_cols]

    if numeric_cols:
        numeric_values = feature_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        # Replace missing numeric values with column medians to keep KMeans stable.
        medians = numeric_values.median().fillna(0)
        numeric_values = numeric_values.fillna(medians)
        scaler = StandardScaler()
        numeric_vals = scaler.fit_transform(numeric_values)
    else:
        numeric_vals = None

    transformed: list[np.ndarray] = []
    if numeric_vals is not None:
        transformed.append(numeric_vals)

    if categorical_cols:
        categoricals = feature_df[categorical_cols].fillna("Unknown")
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # Fallback for older scikit-learn versions
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(categoricals)
        transformed.append(encoded)

    if contextual_features is not None:
        transformed.append(contextual_features)

    if not transformed:
        return df.assign(cluster="Not computed"), None

    matrix = np.hstack(transformed)
    if not np.isfinite(matrix).all():
        return df.assign(cluster="Not computed"), None

    n_init = "auto"
    try:
        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    except TypeError:  # Older scikit-learn compatibility
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
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
        if pd.notnull(min_date_raw) and pd.notnull(max_date_raw):
            recent_start = max_date_raw - pd.DateOffset(months=6)
            if pd.notnull(recent_start):
                bounded_start = max(recent_start, min_date_raw)
                default_range = (bounded_start.date(), max_date)
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
        start_raw, end_raw = state.date_range
        start_ts = pd.to_datetime(start_raw)
        end_ts = pd.to_datetime(end_raw)
        if pd.isna(start_ts) or pd.isna(end_ts):
            start_ts, end_ts = df[DATE_COL].min(), df[DATE_COL].max()
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        filtered = filtered[(filtered[DATE_COL] >= start_ts) & (filtered[DATE_COL] <= end_ts)]
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
# High-level summaries
# ---------------------------------------------------------------------------
def render_filter_summary(filtered: pd.DataFrame, state: FilterState) -> None:
    """Display high-level context about the active filters and dataset."""

    total_events = int(len(filtered))
    fatalities_total = 0
    if FATALITIES_COL in filtered.columns:
        fatalities_total = int(filtered[FATALITIES_COL].fillna(0).sum())

    unique_countries = 0
    if COUNTRY_COL in filtered.columns:
        unique_countries = int(filtered[COUNTRY_COL].dropna().nunique())

    unique_actors = 0
    if ACTOR1_COL in filtered.columns:
        unique_actors = int(filtered[ACTOR1_COL].dropna().nunique())

    date_range_text = ""
    if state.date_range and len(state.date_range) == 2:
        start_ts = pd.to_datetime(state.date_range[0])
        end_ts = pd.to_datetime(state.date_range[1])
        if pd.notnull(start_ts) and pd.notnull(end_ts):
            if start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts
            date_range_text = f"Active date range: {start_ts.date()} to {end_ts.date()}"

    top_country_summary = ""
    if COUNTRY_COL in filtered.columns and total_events:
        top_countries = (
            filtered[COUNTRY_COL]
            .dropna()
            .astype(str)
            .value_counts()
            .head(3)
            .index.tolist()
        )
        if top_countries:
            top_country_summary = "Top countries: " + ", ".join(top_countries)

    event_col, fatality_col, country_col, actor_col = st.columns(4)
    event_col.metric("Events", f"{total_events:,}")
    fatality_col.metric("Reported fatalities", f"{fatalities_total:,}")
    country_col.metric("Countries", f"{unique_countries:,}")
    actor_col.metric("Primary actors", f"{unique_actors:,}")

    caption_parts = [part for part in [date_range_text, top_country_summary] if part]
    if caption_parts:
        st.caption(" · ".join(caption_parts))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_landing_tab():
    st.subheader("Getting started")
    st.markdown(
        """
        Welcome to the ACLED Conflict Analytics Platform. The application preloads the curated
        ACLED dataset bundled with this tool, so you can begin exploring immediately. By default the
        global filters display the latest six months of activity, ensuring the landing view highlights
        the most recent conflict dynamics.

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
    colour_column = color_choice
    if colour_column in map_df.columns and map_df[colour_column].dtype != object:
        colour_column = f"{color_choice}_label"
        map_df[colour_column] = map_df[color_choice].astype(str)
    colour_map = build_colour_mapping(map_df, colour_column, palette_key=color_choice)
    colour_args: dict = {}
    if colour_map:
        colour_args = {
            "color_discrete_map": colour_map,
            "category_orders": {colour_column: list(colour_map.keys())},
        }

    size_args: dict[str, float] = {}
    if size_choice != "None" and size_choice in map_df.columns:
        size_args = {
            "size": map_df[size_choice].clip(lower=0).fillna(0) + 2,
            "size_max": 14,
        }

    fig = px.scatter_mapbox(
        map_df,
        lat=LAT_COL,
        lon=LON_COL,
        color=colour_column,
        hover_data=None,
        custom_data=["event_story"],
        zoom=3,
        height=600,
        **size_args,
        **colour_args,
    )
    marker_style = dict(opacity=0.82)
    if size_choice == "None":
        marker_style["size"] = 6
    marker_style.setdefault("sizemin", 3)
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker=marker_style)
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend=dict(orientation="h", yanchor="bottom", y=0.99, x=0, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
    st.download_button(
        "Download map events",
        data=map_df.drop(columns=["event_story"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="overview_map_events.csv",
        mime="text/csv",
    )

    st.subheader("Temporal trend")
    trend_freq = st.selectbox("Aggregate by", ["week", "month", "year"], index=1)
    trend_df = (
        filtered.groupby(trend_freq)
        .agg(events=("event_id_cnty", "count"), fatalities=(FATALITIES_COL, "sum"))
        .reset_index()
        .sort_values(trend_freq)
    )
    if trend_df.empty:
        st.info("Temporal trend data will appear once events match the selected filters.")
    else:
        line_fig = px.line(trend_df, x=trend_freq, y="events", markers=True)
        line_fig.update_layout(yaxis_title="Number of events", xaxis_title=trend_freq.title())
        st.plotly_chart(line_fig, use_container_width=True)
        st.download_button(
            "Download temporal trend",
            data=trend_df.to_csv(index=False).encode("utf-8"),
            file_name=f"overview_trend_{trend_freq}.csv",
            mime="text/csv",
        )


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
    available_search_cols = [col for col in search_cols if col in filtered.columns]
    keyword_results = filtered[available_search_cols].copy()
    if keyword_results.empty:
        st.warning("No keyword results available for the current filters.")
    else:
        st.dataframe(keyword_results.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download keyword matches",
            data=keyword_results.to_csv(index=False).encode("utf-8"),
            file_name="search_keyword_results.csv",
            mime="text/csv",
        )

    st.subheader("Semantic search (NLP)")
    st.markdown(
        "Describe the type of incident you are investigating to retrieve semantically similar events, even when exact keywords differ."
    )
    semantic_query = st.text_input("Semantic query", placeholder="e.g. attacks on aid workers near border crossings")
    semantic_limit = st.slider("Number of semantic matches", min_value=5, max_value=50, value=15, step=5)
    semantic_results = pd.DataFrame()
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

    if not semantic_results.empty:
        st.download_button(
            "Download semantic matches",
            data=semantic_results.drop(columns=["event_story"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="search_semantic_results.csv",
            mime="text/csv",
        )

    st.subheader("Contextual highlights")
    context_summary = build_context_summary(filtered, context_matrix, context_index, context_vectorizer)
    if context_summary.empty:
        st.info("Contextual themes will appear once there is sufficient narrative information in the filtered events.")
    else:
        st.markdown(
            "The table below surfaces the most salient note themes together with a representative event for each context."
        )
        st.dataframe(context_summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download contextual themes",
            data=context_summary.to_csv(index=False).encode("utf-8"),
            file_name="search_contextual_themes.csv",
            mime="text/csv",
        )

    if filter_state.context_filters:
        context_counts = (
            filtered.assign(match_context=lambda x: x[NOTES_COL].str.lower())
            .assign(
                matches=lambda x: x["match_context"].apply(
                    lambda txt: [ctx for ctx in filter_state.context_filters if ctx.lower() in txt]
                )
            )
        )
        context_filter_summary = (
            context_counts.explode("matches")
            .dropna(subset=["matches"])
            .groupby("matches")
            .agg(events=("event_id_cnty", "count"), fatalities=(FATALITIES_COL, "sum"))
            .reset_index()
            .rename(columns={"matches": "context"})
            .sort_values("events", ascending=False)
        )
        if context_filter_summary.empty:
            st.warning("No contextual matches found in the filtered events.")
        else:
            st.table(context_filter_summary)
            st.download_button(
                "Download context filter summary",
                data=context_filter_summary.to_csv(index=False).encode("utf-8"),
                file_name="search_context_filter_summary.csv",
                mime="text/csv",
            )
    else:
        st.info("Add context terms from the sidebar to analyse themes in the notes column.")


def render_clustering_tab(
    filtered: pd.DataFrame,
    context_matrix,
    context_index: dict,
    context_vectorizer: TfidfVectorizer,
    note_embeddings,
    note_index_lookup: dict | None,
):
    st.subheader("Cluster events by attributes")
    st.markdown(
        "Group events using K-means clustering across spatial, temporal, categorical, and optional narrative features to reveal "
        "cohesive incident archetypes."
    )
    st.caption(
        "When narrative themes are enabled, the notes column is vectorised with TF-IDF, compressed into latent topics via "
        "truncated SVD, and blended with the selected variables before clustering."
    )
    available_features = [LAT_COL, LON_COL, FATALITIES_COL, "year", EVENT_TYPE_COL, SUB_EVENT_COL, ADMIN1_COL]
    selected_features = st.multiselect(
        "Select features for clustering",
        options=available_features,
        default=[LAT_COL, LON_COL, FATALITIES_COL],
    )
    use_context_topics = st.checkbox(
        "Incorporate narrative themes from notes",
        value=False,
        help="Augment clustering with latent themes derived from the notes column to group events with similar narratives.",
    )
    cluster_count = st.slider("Number of clusters", min_value=2, max_value=10, value=4)
    run_cluster = st.button("Run clustering")

    if not run_cluster:
        st.info("Select features and click 'Run clustering' to generate event clusters.")
        return

    if not selected_features:
        st.error("Select at least one feature before running clustering.")
        return

    missing_features = [feature for feature in selected_features if feature not in filtered.columns]
    if missing_features:
        st.error(
            "The following features are unavailable in the filtered dataset: "
            + ", ".join(missing_features)
        )
        return

    contextual_features = None
    svd_model: TruncatedSVD | None = None
    if use_context_topics:
        contextual_features, svd_model = contextual_feature_matrix(
            filtered, context_matrix, context_index
        )
        if contextual_features is None or svd_model is None:
            st.warning("Contextual embeddings could not be generated for the current selection.")
        else:
            topic_terms: list[str] = []
            feature_names = context_vectorizer.get_feature_names_out()
            for component in svd_model.components_[:3]:
                top_term_ids = component.argsort()[::-1][:5]
                topic_terms.append(
                    ", ".join(feature_names[idx] for idx in top_term_ids)
                )
            if topic_terms:
                st.markdown(
                    "**Top narrative themes incorporated:**\n" + "\n".join(
                        f"• {terms}" for terms in topic_terms
                    )
                )

    clustered_df, silhouette = cluster_events(
        filtered,
        selected_features,
        cluster_count,
        contextual_features=contextual_features,
    )
    if "cluster" not in clustered_df.columns or clustered_df["cluster"].isna().all():
        st.warning("Unable to compute clusters with the current selection.")
        return
    if clustered_df["cluster"].eq("Not computed").all():
        st.warning("Clustering could not be computed with the selected configuration.")
        return

    st.success("Clustering complete.")
    if silhouette is not None:
        st.metric("Silhouette score", f"{silhouette:.3f}")
    cluster_counts = clustered_df.groupby("cluster").agg(
        events=("event_id_cnty", "count"),
        mean_fatalities=(FATALITIES_COL, "mean"),
    )
    st.dataframe(cluster_counts, use_container_width=True)
    st.download_button(
        "Download cluster summary",
        data=cluster_counts.to_csv().encode("utf-8"),
        file_name="cluster_summary.csv",
        mime="text/csv",
    )

    cluster_display = clustered_df.assign(event_story=lambda x: x.apply(craft_event_story, axis=1))
    cluster_display["cluster"] = cluster_display["cluster"].astype(str)
    cluster_colour_map = build_colour_mapping(cluster_display, "cluster", palette_key="cluster")
    cluster_colour_args: dict = {}
    if cluster_colour_map:
        cluster_colour_args = {
            "color_discrete_map": cluster_colour_map,
            "category_orders": {"cluster": list(cluster_colour_map.keys())},
        }
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
        **cluster_colour_args,
    )
    cluster_fig.update_traces(
        hovertemplate="Cluster %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
        marker=dict(size=7),
        opacity=0.85,
    )
    cluster_fig.update_layout(mapbox_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(cluster_fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
    st.download_button(
        "Download clustered events",
        data=clustered_df.to_csv(index=False).encode("utf-8"),
        file_name="clustered_events.csv",
        mime="text/csv",
    )

    cluster_theme_summary = summarise_cluster_contexts(
        clustered_df, context_matrix, context_index, context_vectorizer
    )
    if not cluster_theme_summary.empty:
        st.subheader("Narrative themes by cluster")
        st.caption(
            "TF-IDF themes from the notes column are aggregated per cluster to clarify how narratives differ between groups."
        )
        st.dataframe(cluster_theme_summary.sort_values("cluster"), use_container_width=True, hide_index=True)
        st.download_button(
            "Download cluster theme summary",
            data=cluster_theme_summary.to_csv(index=False).encode("utf-8"),
            file_name="cluster_theme_summary.csv",
            mime="text/csv",
        )

    st.subheader("Cluster spotlights")
    st.caption("A concise sample of events from each cluster highlights how the groups differ in practice.")
    for cluster_label, group in clustered_df.groupby("cluster"):
        sample = group.copy()
        if FATALITIES_COL in sample.columns:
            sample = sample.sort_values(FATALITIES_COL, ascending=False)
        sample = sample.head(5)
        if sample.empty:
            continue
        display = sample[[DATE_COL, COUNTRY_COL, ADMIN1_COL, ACTOR1_COL, EVENT_TYPE_COL, NOTES_COL, FATALITIES_COL]].copy()
        display[DATE_COL] = pd.to_datetime(display[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
        display[DATE_COL] = display[DATE_COL].fillna("Unknown")
        display["location"] = (
            display[[ADMIN1_COL, COUNTRY_COL]]
            .astype(str)
            .apply(lambda x: ", ".join([part for part in x if part and part != "nan"]), axis=1)
        )
        display["actor"] = display[ACTOR1_COL].fillna("Unknown actor")
        display["event"] = display[EVENT_TYPE_COL].fillna("Unknown event")
        display["fatalities"] = display[FATALITIES_COL].fillna(0).astype(int)
        display["notes_excerpt"] = display[NOTES_COL].fillna("").apply(
            lambda text: textwrap.shorten(str(text).strip(), width=140, placeholder="…")
        )
        spotlight = display[[DATE_COL, "location", "actor", "event", "fatalities", "notes_excerpt"]]
        st.markdown(f"**Cluster {cluster_label}** — {len(group)} events")
        st.dataframe(spotlight, use_container_width=True, hide_index=True)

    st.subheader("Language-guided narrative clustering")
    st.markdown(
        "Describe the conflict pattern you want to explore and the embedded language model will group the filtered events"
        " around that description using the notes column."
    )
    st.caption(
        "Example instruction: `Cluster events through different rebel groups in the dataset.` The model searches event notes"
        " for the requested pattern before clustering similar narratives."
    )
    if note_embeddings is None or note_index_lookup is None:
        st.info(
            "Language-guided clustering requires the optional `sentence-transformers` dependency. Install it from the"
            " requirements file to enable this feature."
        )
        return

    language_prompt = st.text_area(
        "Describe the pattern to cluster",
        key="language_cluster_prompt",
        placeholder="Cluster events through different rebel groups in the dataset.",
        help="Explain the narrative focus. The notes column will be searched for matches before clustering similar events.",
    )
    language_cluster_count = st.slider(
        "Number of language-guided clusters",
        min_value=2,
        max_value=6,
        value=3,
        key="language_cluster_count",
    )
    language_event_limit = st.slider(
        "Maximum events to analyse",
        min_value=60,
        max_value=400,
        value=200,
        step=20,
        key="language_event_limit",
        help="Limits how many of the most relevant events are clustered to keep the narrative view focused.",
    )
    run_language_cluster = st.button(
        "Generate language-guided clusters",
        key="language_cluster_button",
    )

    if run_language_cluster:
        result_df, status = language_guided_clustering(
            filtered,
            note_embeddings,
            note_index_lookup,
            language_prompt,
            language_cluster_count,
            max_events=language_event_limit,
        )
        if status == "model_unavailable":
            st.error("Language model embeddings are unavailable. Ensure `sentence-transformers` is installed.")
            return
        if status == "empty_query":
            st.warning("Provide an instruction so the language model knows what pattern to search for.")
            return
        if status == "no_embeddings":
            st.warning("No narrative text is available for the filtered events, so language-guided clustering cannot run.")
            return
        if status == "no_similarity":
            st.warning("The description did not match any event notes. Refine the instruction and try again.")
            return
        if status == "insufficient_events":
            st.warning("Not enough events matched the description to form distinct clusters. Adjust the filters or prompt.")
            return

        st.success("Language-guided clustering complete.")
        result_df = result_df.sort_values("similarity", ascending=False)
        language_summary = result_df.groupby("cluster").agg(
            events=("event_id_cnty", "count"),
            mean_similarity=("similarity", "mean"),
            mean_fatalities=(FATALITIES_COL, "mean"),
        )
        st.dataframe(language_summary, use_container_width=True)

        theme_summary = summarise_cluster_contexts(
            result_df,
            context_matrix,
            context_index,
            context_vectorizer,
        )
        if not theme_summary.empty:
            st.caption("Narrative keywords for each language-guided cluster:")
            st.table(theme_summary.sort_values("cluster"))

        st.caption("Representative events per language-guided cluster:")
        for cluster_label, group in result_df.groupby("cluster"):
            cluster_sample = group.sort_values("similarity", ascending=False).head(5)
            display = cluster_sample[
                [DATE_COL, COUNTRY_COL, ADMIN1_COL, ACTOR1_COL, EVENT_TYPE_COL, NOTES_COL, FATALITIES_COL, "similarity"]
            ].copy()
            display[DATE_COL] = pd.to_datetime(display[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
            display[DATE_COL] = display[DATE_COL].fillna("Unknown")
            display["location"] = (
                display[[ADMIN1_COL, COUNTRY_COL]]
                .astype(str)
                .apply(lambda x: ", ".join([part for part in x if part and part != "nan"]), axis=1)
            )
            display["actor"] = display[ACTOR1_COL].fillna("Unknown actor")
            display["event"] = display[EVENT_TYPE_COL].fillna("Unknown event")
            display["fatalities"] = display[FATALITIES_COL].fillna(0).astype(int)
            display["similarity"] = display["similarity"].round(3)
            display["notes_excerpt"] = display[NOTES_COL].fillna("").apply(
                lambda text: textwrap.shorten(str(text).strip(), width=140, placeholder="…")
            )
            table = display[
                [DATE_COL, "location", "actor", "event", "fatalities", "similarity", "notes_excerpt"]
            ]
            st.markdown(f"**Language cluster {cluster_label}** — {len(group)} events")
            st.dataframe(table, use_container_width=True, hide_index=True)


def render_network_tab(filtered: pd.DataFrame):
    st.subheader("Actor association network")
    st.markdown("Construct a co-occurrence network from primary actors and their associated actors within events.")
    st.caption(
        "Nodes represent actors; edges are weighted by the number of events in which the pair co-occur. Node size follows"
        " weighted degree centrality while node colour encodes the fatalities attributed to each actor across the filtered"
        " events."
    )
    event_count = len(filtered)
    if event_count < 10:
        st.info(
            f"At least 10 events are required to render the actor network. Only {event_count} events match the current filters."
        )
        return

    actor_values = pd.concat(
        [
            filtered.get(ACTOR1_COL, pd.Series(dtype=str)).fillna("").astype(str).str.strip(),
            filtered.get(ASSOC_ACTOR1_COL, pd.Series(dtype=str)).fillna("").astype(str).str.strip(),
        ],
        ignore_index=True,
    )
    actor_values = actor_values[actor_values != ""]
    if actor_values.empty or actor_values.value_counts().max() <= 1:
        st.warning(
            "Actor overlaps are required to form edges, but no actor appears in more than one association within the selected"
            " filters."
        )
        return

    graph = build_actor_network(filtered)
    if graph.number_of_edges() == 0:
        st.warning("Not enough actor association data to build a network for the selected filters.")
        return

    degree_series = pd.Series(dict(graph.degree(weight="weight")))
    if degree_series.empty:
        st.warning("Unable to determine actor centrality for the selected filters.")
        return
    max_nodes_available = int(degree_series.size)
    if max_nodes_available < 2:
        st.warning("Not enough distinct actors to build a network for the selected filters.")
        return
    slider_max = max_nodes_available
    slider_min = max(2, min(5, slider_max))
    if slider_min >= slider_max:
        top_n = slider_max
        st.caption(
            f"Showing all {top_n} actors because the filtered dataset exposes only a limited number of participants."
        )
    else:
        slider_default = min(10, slider_max)
        top_n = st.slider(
            "Show top actors",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
        )
    top_nodes = degree_series.sort_values(ascending=False).head(top_n).index.tolist()
    subgraph = graph.subgraph(top_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        st.warning("No actor relationships available after applying the selected filters.")
        return
    centrality = nx.degree_centrality(subgraph)
    weighted_degree = dict(subgraph.degree(weight="weight"))
    pos = nx.kamada_kawai_layout(subgraph, weight="weight")

    edge_traces: list[go.Scatter] = []
    for src, dst, attrs in subgraph.edges(data=True):
        weight = attrs.get("weight", 1)
        edge_trace = go.Scatter(
            x=[pos[src][0], pos[dst][0]],
            y=[pos[src][1], pos[dst][1]],
            mode="lines",
            line=dict(width=0.8 + weight * 0.15, color="#bcd2e8"),
            hoverinfo="text",
            text=f"{src} ↔ {dst}<br>Interactions: {weight}",
        )
        edge_traces.append(edge_trace)

    node_order = list(subgraph.nodes())
    degree_values = np.array([weighted_degree.get(node, 0) for node in node_order], dtype=float)
    fatality_values = np.array(
        [subgraph.nodes[node].get("fatalities", 0.0) for node in node_order], dtype=float
    )
    event_totals = np.array([subgraph.nodes[node].get("events", 0) for node in node_order], dtype=float)
    if degree_values.size == 0:
        node_sizes = []
        node_colours = []
        colour_max = 1
    else:
        max_degree = degree_values.max()
        min_size, max_size = 14, 34
        if max_degree == 0:
            scaled = np.zeros_like(degree_values)
        else:
            scaled = np.sqrt(degree_values / max_degree)
        node_sizes = (min_size + (max_size - min_size) * scaled).tolist()
        node_colours = fatality_values.tolist()
        colour_max = float(fatality_values.max()) if fatality_values.size else 1
        if colour_max <= 0:
            colour_max = 1
    node_customdata = np.column_stack(
        [
            [weighted_degree.get(node, 0) for node in node_order],
            [centrality.get(node, 0.0) for node in node_order],
            fatality_values,
            event_totals,
        ]
    )
    node_trace = go.Scatter(
        x=[pos[node][0] for node in node_order],
        y=[pos[node][1] for node in node_order],
        mode="markers+text",
        text=node_order,
        textposition="top center",
        textfont=dict(size=12, color="#1f2937"),
        marker=dict(
            size=node_sizes,
            color=node_colours,
            colorscale="Reds",
            cmin=0,
            cmax=colour_max,
            showscale=True,
            colorbar=dict(title="Attributed fatalities"),
            line=dict(width=1.2, color="#f8fafc"),
        ),
        customdata=node_customdata,
        hovertemplate=(
            "<b>%{text}</b><br>Weighted degree: %{customdata[0]:.0f}"
            "<br>Centrality: %{customdata[1]:.3f}"
            "<br>Attributed fatalities: %{customdata[2]:.0f}"
            "<br>Events observed: %{customdata[3]:.0f}<extra></extra>"
        ),
    )

    network_fig = go.Figure(data=edge_traces + [node_trace])
    network_fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        title=dict(text="Actor co-occurrence network", x=0.5, font=dict(size=16, color="#1f2937")),
    )
    st.plotly_chart(network_fig, use_container_width=True, config={"displayModeBar": False})

    centrality_df = (
        pd.DataFrame(
            {
                "actor": list(subgraph.nodes()),
                "weighted_degree": [weighted_degree.get(node, 0) for node in subgraph.nodes()],
                "centrality": [centrality.get(node, 0) for node in subgraph.nodes()],
                "attributed_fatalities": [subgraph.nodes[node].get("fatalities", 0) for node in subgraph.nodes()],
                "events_observed": [subgraph.nodes[node].get("events", 0) for node in subgraph.nodes()],
            }
        )
        .sort_values("centrality", ascending=False)
        .reset_index(drop=True)
    )
    st.table(centrality_df)
    st.download_button(
        "Download network centrality",
        data=centrality_df.to_csv(index=False).encode("utf-8"),
        file_name="network_centrality.csv",
        mime="text/csv",
    )

    edge_df = (
        pd.DataFrame(
            [
                {"source": src, "target": dst, "weight": attrs.get("weight", 1)}
                for src, dst, attrs in subgraph.edges(data=True)
            ]
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    if not edge_df.empty:
        st.download_button(
            "Download network edges",
            data=edge_df.to_csv(index=False).encode("utf-8"),
            file_name="network_edges.csv",
            mime="text/csv",
        )


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
    note_embeddings, note_index_lookup = build_note_embeddings(data[NOTES_COL])

    filter_state = render_sidebar(data)
    filtered = apply_filters(data, filter_state)

    st.success(f"Filtered dataset contains {len(filtered):,} events")
    render_filter_summary(filtered, filter_state)

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
        render_clustering_tab(
            filtered,
            context_matrix,
            context_index,
            context_vectorizer,
            note_embeddings,
            note_index_lookup,
        )
    with network_tab:
        render_network_tab(filtered)
    with data_tab:
        render_data_tab(filtered)


if __name__ == "__main__":
    main()

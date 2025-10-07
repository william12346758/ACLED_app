# -*- coding: utf-8 -*-
"""Streamlit analytic tool for ACLED data.

Created on Mon Oct  6 15:38:27 2025
Author: LWu
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
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


class DataLoadingError(RuntimeError):
    """Raised when the ACLED source dataset cannot be loaded."""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "ACLED 2016-2025.xlsx"
CACHE_PATH = Path(__file__).resolve().parent / "acled_dataset.pkl"
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
INTER1_COL = "inter1"


DATA_USE_COLUMNS = [
    "event_id_cnty",
    DATE_COL,
    EVENT_TYPE_COL,
    SUB_EVENT_COL,
    ACTOR1_COL,
    ASSOC_ACTOR1_COL,
    INTER1_COL,
    COUNTRY_COL,
    ADMIN1_COL,
    LAT_COL,
    LON_COL,
    NOTES_COL,
    FATALITIES_COL,
]

st.set_page_config(page_title="ACLED Conflict Analytics", layout="wide")


logger = logging.getLogger(__name__)


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
REQUIRED_COLUMNS = {DATE_COL, LAT_COL, LON_COL, NOTES_COL}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ACLED dataframe with consistent typing and derived columns."""

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise DataLoadingError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prepared = df.copy()
    prepared[DATE_COL] = pd.to_datetime(prepared[DATE_COL], errors="coerce")
    prepared[FATALITIES_COL] = pd.to_numeric(prepared[FATALITIES_COL], errors="coerce")
    prepared[LAT_COL] = pd.to_numeric(prepared[LAT_COL], errors="coerce")
    prepared[LON_COL] = pd.to_numeric(prepared[LON_COL], errors="coerce")
    prepared = prepared.dropna(subset=[DATE_COL, LAT_COL, LON_COL])
    prepared["year"] = prepared[DATE_COL].dt.year
    prepared["month"] = prepared[DATE_COL].dt.strftime("%Y-%m")
    prepared["week"] = prepared[DATE_COL].dt.strftime("%Y-%W")
    prepared[NOTES_COL] = prepared[NOTES_COL].fillna("")
    prepared[ACTOR1_COL] = prepared[ACTOR1_COL].fillna("Unknown actor")
    prepared[ASSOC_ACTOR1_COL] = prepared[ASSOC_ACTOR1_COL].fillna("")
    if INTER1_COL in prepared.columns:
        prepared[INTER1_COL] = prepared[INTER1_COL].fillna("Unknown category")
    return prepared


def _csv_candidates() -> list[Path]:
    """Return ordered CSV fallbacks located alongside the app."""

    parent = DATA_PATH.parent
    primary_csv = DATA_PATH.with_suffix(".csv")
    candidates: list[Path] = []
    seen: set[Path] = set()

    for path in [primary_csv, *sorted(parent.glob("*.csv"))]:
        if path.exists() and path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates


def _load_cached_dataframe() -> pd.DataFrame | None:
    """Return the cached ACLED dataframe if it is fresher than the source files."""

    if not CACHE_PATH.exists():
        return None

    try:
        cached = pd.read_pickle(CACHE_PATH)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to read cached dataset %s: %s", CACHE_PATH.name, exc)
        return None

    if not isinstance(cached, pd.DataFrame):
        logger.warning("Cached dataset %s is not a DataFrame.", CACHE_PATH.name)
        return None

    source_candidates = [DATA_PATH, *_csv_candidates()]
    freshest_source_mtime: float | None = None
    for candidate in source_candidates:
        if candidate.exists():
            try:
                freshest_source_mtime = max(
                    freshest_source_mtime or 0.0, candidate.stat().st_mtime
                )
            except OSError:
                continue

    try:
        cache_mtime = CACHE_PATH.stat().st_mtime
    except OSError:
        return None

    if freshest_source_mtime is None or cache_mtime >= freshest_source_mtime:
        return cached.copy()

    return None


def _persist_cache(df: pd.DataFrame) -> None:
    try:
        df.to_pickle(CACHE_PATH)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to update dataset cache %s: %s", CACHE_PATH.name, exc)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load ACLED data from disk with typed columns."""

    cached = _load_cached_dataframe()
    if cached is not None:
        return cached

    errors: list[str] = []
    openpyxl_reported = False

    for source_path in [DATA_PATH, *_csv_candidates()]:
        if not source_path.exists():
            continue

        try:
            usecols = list(dict.fromkeys(DATA_USE_COLUMNS))
            if source_path.suffix.lower() == ".xlsx":
                raw = pd.read_excel(source_path, usecols=usecols)
            else:
                raw = pd.read_csv(source_path, usecols=usecols)
            prepared = _prepare_dataframe(raw)
            _persist_cache(prepared)
            return prepared
        except ImportError:
            if not openpyxl_reported:
                errors.append(
                    "Excel dataset present but optional dependency 'openpyxl' is not installed. "
                    "Install it with `pip install openpyxl` or provide a CSV export instead."
                )
                openpyxl_reported = True
        except DataLoadingError as exc:
            errors.append(f"{source_path.name} is incompatible: {exc}")
        except Exception as exc:  # pragma: no cover - defensive guard
            errors.append(f"Unable to read {source_path.name}: {exc}")

    error_message = "Unable to load the ACLED dataset."
    if errors:
        unique_errors = []
        for message in errors:
            if message not in unique_errors:
                unique_errors.append(message)
        error_message += "\n- " + "\n- ".join(unique_errors)
    raise DataLoadingError(error_message)


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


def filter_by_terms(
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
        actor_type = str(row.get(INTER1_COL) or "").strip()
        if not actor or not actor_type or actor == actor_type:
            continue
        fatalities_value = row.get(FATALITIES_COL)
        fatality = float(fatalities_value) if pd.notnull(fatalities_value) else 0.0
        for node, node_type in ((actor, "Actor"), (actor_type, "Actor category")):
            if not graph.has_node(node):
                graph.add_node(node, fatalities=0.0, events=0, node_type=node_type)
            else:
                graph.nodes[node].setdefault("node_type", node_type)
            graph.nodes[node]["fatalities"] += fatality
            graph.nodes[node]["events"] += 1
        if graph.has_edge(actor, actor_type):
            graph[actor][actor_type]["weight"] += 1
        else:
            graph.add_edge(actor, actor_type, weight=1)
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
    text_filters: list[str]
    text_match_all: bool
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
        st.subheader("Text search")
        text_raw = st.text_input(
            "Keywords or phrases (comma separated)",
            help="Search across notes, actor names, and location text using any comma-separated terms.",
        )
        text_logic = st.selectbox(
            "Text match",
            ["Match any", "Match all"],
            index=0,
            help="Match any will return events containing at least one term; Match all requires every term to appear.",
        )

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
        text_filters=parse_keywords(text_raw),
        text_match_all=text_logic == "Match all",
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
    filtered = filter_by_terms(
        filtered,
        state.text_filters,
        [NOTES_COL, ACTOR1_COL, ASSOC_ACTOR1_COL, ADMIN1_COL, "location"],
        state.text_match_all,
    )
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

    fig = px.scatter_map(
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
        map_style="carto-positron",
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
):
    st.subheader("Text search results")
    if filter_state.text_filters:
        st.write("Matching events for search terms:")
        st.code(", ".join(filter_state.text_filters), language="text")
    else:
        st.info("Add text filters from the sidebar to search across notes, actors, and locations.")

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
        st.dataframe(keyword_results.head(200), width="stretch", hide_index=True)
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
                width="stretch",
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

    # Contextual highlights removed per user feedback.


def render_clustering_tab(
    filtered: pd.DataFrame,
    context_matrix,
    context_index: dict,
    context_vectorizer: TfidfVectorizer,
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
    st.dataframe(cluster_counts, width="stretch")
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
    cluster_fig = px.scatter_map(
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
    cluster_fig.update_layout(map_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
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
        st.dataframe(cluster_theme_summary.sort_values("cluster"), width="stretch", hide_index=True)
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
        st.dataframe(spotlight, width="stretch", hide_index=True)

def render_network_tab(filtered: pd.DataFrame):
    st.subheader("Actor-category network")
    st.markdown(
        "Construct a bipartite network linking primary actors (`actor1`) with their ACLED actor categories (`inter1`)."
    )
    st.caption(
        "Nodes represent either specific actors or actor categories. Edges are weighted by the number of events"
        " connecting the pair. Node size reflects weighted degree centrality while colour encodes the fatalities"
        " attributed to that actor or category across the filtered events."
    )
    event_count = len(filtered)
    if event_count < 10:
        st.info(
            f"At least 10 events are required to render the actor network. Only {event_count} events match the current filters."
        )
        return

    pair_df = filtered[[ACTOR1_COL, INTER1_COL, FATALITIES_COL]].copy()
    pair_df[ACTOR1_COL] = pair_df[ACTOR1_COL].fillna("").astype(str).str.strip()
    pair_df[INTER1_COL] = pair_df[INTER1_COL].fillna("").astype(str).str.strip()
    pair_df = pair_df[(pair_df[ACTOR1_COL] != "") & (pair_df[INTER1_COL] != "")]
    if pair_df.empty:
        st.warning(
            "Primary actor names and actor categories are required to generate the network, but the filtered events lack this information."
        )
        return

    graph = build_actor_network(pair_df)
    if graph.number_of_edges() == 0:
        st.warning("Not enough actor-category links are available to build a network for the selected filters.")
        return

    degree_series = pd.Series(dict(graph.degree(weight="weight")))
    if degree_series.empty:
        st.warning("Unable to determine network centrality for the selected filters.")
        return
    max_nodes_available = int(degree_series.size)
    if max_nodes_available < 2:
        st.warning("Not enough distinct entities to build a network for the selected filters.")
        return
    slider_max = max_nodes_available
    slider_min = max(2, min(5, slider_max))
    if slider_min >= slider_max:
        top_n = slider_max
        st.caption(
            f"Showing all {top_n} entities because the filtered dataset exposes only a limited number of participants."
        )
    else:
        slider_default = min(10, slider_max)
        top_n = st.slider(
            "Show top entities",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
        )
    top_nodes = degree_series.sort_values(ascending=False).head(top_n).index.tolist()
    subgraph = graph.subgraph(top_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        st.warning("No actor-category relationships available after applying the selected filters.")
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
            text=f"{src} ↔ {dst}<br>Events: {weight}",
        )
        edge_traces.append(edge_trace)

    node_order = list(subgraph.nodes())
    degree_values = np.array([weighted_degree.get(node, 0) for node in node_order], dtype=float)
    fatality_values = np.array(
        [subgraph.nodes[node].get("fatalities", 0.0) for node in node_order], dtype=float
    )
    event_totals = np.array([subgraph.nodes[node].get("events", 0) for node in node_order], dtype=float)
    centrality_values = np.array([centrality.get(node, 0.0) for node in node_order], dtype=float)
    node_types = np.array([subgraph.nodes[node].get("node_type", "Actor") for node in node_order], dtype=object)
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
            degree_values,
            centrality_values,
            fatality_values,
            event_totals,
            node_types,
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
            "<b>%{text}</b><br>Role: %{customdata[4]}"
            "<br>Weighted degree: %{customdata[0]:.0f}"
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
        title=dict(text="Actor-category interaction network", x=0.5, font=dict(size=16, color="#1f2937")),
    )
    st.plotly_chart(network_fig, use_container_width=True, config={"displayModeBar": False})

    centrality_df = (
        pd.DataFrame(
            {
                "entity": list(subgraph.nodes()),
                "entity_type": [subgraph.nodes[node].get("node_type", "Actor") for node in subgraph.nodes()],
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

    edge_rows: list[dict] = []
    for src, dst, attrs in subgraph.edges(data=True):
        weight = attrs.get("weight", 1)
        src_type = subgraph.nodes[src].get("node_type", "Actor")
        dst_type = subgraph.nodes[dst].get("node_type", "Actor")
        if src_type == "Actor" and dst_type != "Actor":
            actor_name, category_name = src, dst
        elif dst_type == "Actor" and src_type != "Actor":
            actor_name, category_name = dst, src
        else:
            actor_name, category_name = src, dst
        edge_rows.append(
            {
                "actor": actor_name,
                "actor_category": category_name,
                "events": weight,
            }
        )
    edge_df = (
        pd.DataFrame(edge_rows)
        .sort_values("events", ascending=False)
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
    st.dataframe(filtered, width="stretch")
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

    try:
        data = load_data()
    except DataLoadingError as exc:
        st.error("Unable to load the ACLED dataset.")
        st.markdown(
            "Please ensure that the Excel file is accompanied by the optional `openpyxl` "
            "dependency or place a CSV export of the dataset alongside the app file.\n\n"
            f"**Details:** {exc}"
        )
        st.stop()

    semantic_vectorizer, semantic_matrix, semantic_index = build_semantic_index(data[NOTES_COL])
    context_vectorizer, context_matrix, context_index, _ = build_context_matrix(data[NOTES_COL])

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
        )
    with cluster_tab:
        render_clustering_tab(
            filtered,
            context_matrix,
            context_index,
            context_vectorizer,
        )
    with network_tab:
        render_network_tab(filtered)
    with data_tab:
        render_data_tab(filtered)


if __name__ == "__main__":
    main()

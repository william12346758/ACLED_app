# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:34:11 2025

@author: LWu
"""

import os, math, re
from datetime import datetime
from dateutil import parser as dateparser

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx

import spacy
from spacy.matcher import PhraseMatcher, Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Streamlit config
# --------------------------
st.set_page_config(page_title="ACLED + spaCy — Quick Insights (v2)", layout="wide")

# --------------------------
# Column names (edit to fit your CSV)
# --------------------------
EVENT_ID_COL = "event_id_cnty"
DATE_COL     = "event_date"
COUNTRY_COL  = "country"
ADMIN1_COL   = "admin1"
EVTYPE_COL   = "event_type"
SUBEV_COL    = "sub_event_type"
LAT_COL, LON_COL = "latitude", "longitude"
TEXT_COL     = "notes"
FATAL_COL    = "fatalities"
ACTOR1_COL   = "actor1"

# --------------------------
# Utils
# --------------------------
def coerce_date(s):
    try:
        return dateparser.parse(str(s), dayfirst=False, yearfirst=False)
    except Exception:
        return pd.NaT

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a)) if not np.isnan(a) else np.nan

def normalize_str(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.lower().strip()
    return "".join(ch for ch in x if ch.isalnum() or ch.isspace())

# --------------------------
# Domain lexicons
# --------------------------
ACTION_TERMS = [
    # kinetic
    "attack","ambush","clash","raid","shell","bomb","detonate","shoot","kill","execute",
    # coercion/PD
    "burn","arson","destroy","loot",
    # policing / SD
    "arrest","detain","release","meet","negotiate","announce","deploy","withdraw","evacuate",
    # protest / crowd control
    "protest","demonstrate","riot","block","blockade","strike","disperse","fire"
]
TARGETS = [
    "road","checkpoint","market","school","hospital","bridge","shop","business","police","military",
    "idp camp","camp","teachers","civilians","protesters","protestors","monusco","fardc","m23"
]
WEAPONS = ["ied","grenade","mortar","artillery","explosive","rpg","gun","rifle"]

AUX_VERBS = {"be","have","do","get"}
REPORTING_VERBS = {"say","tell","claim","report","allege","announce","state"}
STOP_VERBS = AUX_VERBS | REPORTING_VERBS | {"make","take","go","come","give","receive"}

VERB_CANON = {
    # kinetic
    "attack":"attack","ambush":"ambush","clash":"clash","raid":"raid",
    "shell":"shell","bomb":"bomb","detonate":"bomb","explode":"bomb","blast":"bomb",
    "shoot":"shoot","kill":"kill","execute":"kill",
    # coercion / PD
    "burn":"arson","arson":"arson","destroy":"destroy","loot":"loot",
    # police / SD
    "arrest":"arrest","detain":"arrest","release":"release",
    "meet":"meet","negotiate":"negotiate","talk":"negotiate","discuss":"negotiate",
    "deploy":"deploy","withdraw":"withdraw","retreat":"withdraw","evacuate":"evacuate",
    "summon":"summon",
    # protest
    "protest":"protest","demonstrate":"protest","riot":"riot","block":"block",
    "blockade":"blockade","strike":"strike","disperse":"disperse","fire":"fire",
}

def canonical_verb(lemma: str) -> str:
    lemma = (lemma or "").lower().strip()
    if not lemma or lemma in STOP_VERBS:
        return ""
    return VERB_CANON.get(lemma, lemma)

# Counts like "arrested 33", "killed 4"
RE_ARREST = re.compile(r"\barrest(?:ed)?\s+(\d{1,4})\b", re.I)
RE_KILLED = re.compile(r"\b(?:shot\s+and\s+killed|killed)\s+(\d{1,4})\b", re.I)
RE_DESTROY = re.compile(r"\bdestroy(?:ed)?\s+(\d{1,4})\b", re.I)
RE_BLOCK_ROAD = re.compile(r"\b(block(?:ed|ade)?)\b.*\broad\b", re.I)

# --------------------------
# spaCy loaders & matchers
# --------------------------
@st.cache_resource(show_spinner=False)
def load_spacy(model_name: str):
    return spacy.load(model_name)

@st.cache_resource(show_spinner=False)
def build_matchers(_nlp, model_name: str):
    phrase = PhraseMatcher(_nlp.vocab, attr="LOWER")
    phrase.add("ACTION_LEX",  [_nlp.make_doc(t) for t in ACTION_TERMS])
    phrase.add("TARGET_LEX",  [_nlp.make_doc(t) for t in TARGETS])
    phrase.add("WEAPON_LEX",  [_nlp.make_doc(t) for t in WEAPONS])
    return phrase

def anchor_verb(sent):
    root = sent.root
    if root.pos_ == "VERB" and root.lemma_ not in STOP_VERBS and root.tag_ != "AUX":
        return root
    for rel in ("xcomp","conj","advcl","ccomp"):
        for tok in root.children:
            if tok.dep_ == rel and tok.pos_ == "VERB" and tok.lemma_ not in STOP_VERBS and tok.tag_ != "AUX":
                return tok
    for tok in sent:
        if tok.pos_ == "VERB" and tok.lemma_ not in STOP_VERBS and tok.tag_ != "AUX":
            return tok
    return None

def extract_microevents(doc, phrase):
    rows = []
    for sent in doc.sents:
        verb = anchor_verb(sent)
        if not verb:
            # try to infer a block-ROAD case from tokens even without a clean verb anchor
            if RE_BLOCK_ROAD.search(sent.text):
                rows.append({
                    "sentence": sent.text, "action_lemma":"block", "agent_guess":"", "target_guess":"road",
                    "negated": False, "lex_hits":["road"], "kv_arrest":None, "kv_killed":None, "kv_destroy":None,
                    "flags":{"road_block":True}
                })
            continue

        a_lemma = canonical_verb(verb.lemma_)
        if not a_lemma:
            continue

        agent = " ".join(w.text for w in verb.lefts if w.dep_ in ("nsubj","nsubjpass"))
        dobj  = [w for w in verb.rights if w.dep_ in ("dobj","attr","dative")]
        pobj  = [c for w in verb.rights if w.dep_=="prep" for c in w.rights if c.dep_=="pobj"]
        target = " ".join(w.text for w in (dobj + pobj))

        hits = []
        for _, s, e in phrase(sent):
            hits.append(sent[s:e].text.lower())

        txt = sent.text
        kv_arrest = m.group(1) if (m := RE_ARREST.search(txt)) else None
        kv_killed = m.group(1) if (m := RE_KILLED.search(txt)) else None
        kv_destroy = m.group(1) if (m := RE_DESTROY.search(txt)) else None

        rows.append({
            "sentence": sent.text,
            "action_lemma": a_lemma,
            "agent_guess": agent,
            "target_guess": target,
            "negated": any(ch.dep_=="neg" for ch in verb.children),
            "lex_hits": sorted(set(hits)),
            "kv_arrest": kv_arrest,
            "kv_killed": kv_killed,
            "kv_destroy": kv_destroy,
            "flags": {"road_block": bool(RE_BLOCK_ROAD.search(txt))}
        })
    return rows

def micro_label(row, event_type_hint:str=""):
    a = row.get("action_lemma","")
    hits = " ".join(row.get("lex_hits", [])).lower()
    sent = row.get("sentence","").lower()

    # high-precision labels for your examples
    if a in {"clash"} or ("clashed" in sent):
        return "Clash"
    if a in {"arrest"} or row.get("kv_arrest"):
        return "Arrest"
    if a in {"kill","shoot"} or row.get("kv_killed"):
        # if civilians indicated
        if re.search(r"\b(civilian|teacher|shopkeeper|business|student|farmer|cleric)\b", sent):
            return "Killing_Civilian"
        return "Killing"
    if a in {"destroy"} or "property destruction" in sent or row.get("kv_destroy"):
        return "Property_Destruction"
    if a in {"block","blockade"} or row.get("flags",{}).get("road_block", False):
        return "Protest_Road_Block"
    if a in {"protest","riot","strike"}:
        return a.capitalize()
    if a in {"meet","negotiate"}:
        return "Meeting/Negotiation"
    if a in {"deploy","withdraw","evacuate","release"}:
        return a.capitalize()
    # fallback using event type
    if "strategic developments" in event_type_hint.lower():
        return "Strategic_Dev"
    return "Other"

def top_verb_dict(series, k=6):
    vc = (series[series.str.len() > 0]).value_counts().head(k)
    return {k: int(v) for k, v in vc.items()}

# --------------------------
# UI — load CSV
# --------------------------
st.title("ACLED Quick Insights (spaCy v2, refined rules)")

up = st.sidebar.file_uploader("Upload ACLED CSV", type=["csv"])
model_choice = st.sidebar.selectbox("spaCy model", ["en_core_web_md","en_core_web_trf","en_core_web_sm"], index=0)

if up is None:
    st.info("Upload a CSV with columns like: "
            f"`{EVENT_ID_COL}, {DATE_COL}, {COUNTRY_COL}, {ADMIN1_COL}, {EVTYPE_COL}, {SUBEV_COL}, {LAT_COL}, {LON_COL}, {TEXT_COL}`")
    st.stop()

df = pd.read_csv(up, dtype=str)
for c in [LAT_COL, LON_COL, FATAL_COL]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
if DATE_COL in df.columns:
    df[DATE_COL] = df[DATE_COL].apply(coerce_date)
else:
    st.error(f"Missing `{DATE_COL}` column."); st.stop()

# de-dup identical narratives (ACLED often has actor splits)
df["text"] = df.get(TEXT_COL, "").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
df = df.drop_duplicates(subset=[EVENT_ID_COL, "text"]).reset_index(drop=True)

df["year"]  = df[DATE_COL].dt.year
df["month"] = df[DATE_COL].dt.to_period("M").astype(str)

years  = sorted([int(y) for y in df["year"].dropna().unique()])
a1opts = sorted(df.get(ADMIN1_COL, pd.Series(dtype=str)).dropna().unique().tolist())
etopts = sorted(df.get(EVTYPE_COL, pd.Series(dtype=str)).dropna().unique().tolist())

y_sel  = st.sidebar.multiselect("Year", years, default=years[-5:] if years else [])
a1_sel = st.sidebar.multiselect("Admin1", a1opts, default=a1opts[:10] if a1opts else [])
et_sel = st.sidebar.multiselect("Event Type", etopts, default=etopts)

mask = pd.Series(True, index=df.index)
if y_sel:  mask &= df["year"].isin(y_sel)
if a1_sel: mask &= df[ADMIN1_COL].isin(a1_sel)
if et_sel: mask &= df[EVTYPE_COL].isin(et_sel)
df_f = df[mask].copy()

st.success(f"Loaded {len(df):,} rows → {len(df_f):,} after filters.")

# --------------------------
# spaCy and matchers
# --------------------------
with st.spinner(f"Loading spaCy `{model_choice}`…"):
    nlp = load_spacy(model_choice)
phrase = build_matchers(nlp, model_choice)

# --------------------------
# Parse a capped set for speed
# --------------------------
N_MAX = st.sidebar.number_input("Max events to parse", min_value=50, max_value=30000, value=min(4000, len(df_f)))
parse_df = df_f.head(int(N_MAX)).copy()

with st.spinner(f"Parsing {len(parse_df):,} narratives…"):
    micro_rows = []
    for r in parse_df.itertuples():
        doc = nlp(getattr(r, "text") or "")
        recs = extract_microevents(doc, phrase)
        for m in recs:
            m["event_id"] = getattr(r, EVENT_ID_COL)
            m["year"] = getattr(r, "year")
            m["country"] = getattr(r, COUNTRY_COL)
            m["admin1"] = getattr(r, ADMIN1_COL)
            m["event_type"] = getattr(r, EVTYPE_COL)
            m["sub_event_type"] = getattr(r, SUBEV_COL)
            m["latitude"] = getattr(r, LAT_COL)
            m["longitude"] = getattr(r, LON_COL)
            m["actor1"] = getattr(r, ACTOR1_COL) if ACTOR1_COL in parse_df.columns else None
            micro_rows.append(m)

micro = pd.DataFrame(micro_rows)
if micro.empty:
    st.warning("No micro-events extracted."); st.stop()

# micro labels with event-type hint
micro["micro_type"] = micro.apply(lambda x: micro_label(x, event_type_hint=str(x.get("event_type",""))), axis=1)

# pull key-value extractions into numeric
for k in ["kv_arrest","kv_killed","kv_destroy"]:
    micro[k] = pd.to_numeric(micro[k], errors="coerce")

# --------------------------
# KPIs
# --------------------------
colA, colB, colC, colD = st.columns(4)
with colA: st.metric("Events (filtered)", f"{df_f[EVENT_ID_COL].nunique():,}")
with colB: st.metric("Parsed sentences", f"{len(micro):,}")
with colC: st.metric("Unique verbs", f"{micro['action_lemma'].nunique():,}")
with colD: st.metric("Unique micro-types", f"{micro['micro_type'].nunique():,}")

# --------------------------
# Partitioned summaries (year × admin1 × event_type)
# --------------------------
st.subheader("Partition summaries (year × admin1 × event_type)")
agg = (micro.groupby(["year","admin1","event_type"])
       .agg(
           events=("event_id","nunique"),
           top_verbs=("action_lemma", lambda s: top_verb_dict(s, 6)),
           micro_types=("micro_type", lambda s: s.value_counts().head(6).to_dict()),
           arrests_sum=("kv_arrest","sum"),
           killed_sum=("kv_killed","sum"),
           destroyed_sum=("kv_destroy","sum")
       ).reset_index())
st.dataframe(agg, use_container_width=True)

# --------------------------
# Event map
# --------------------------
st.subheader("Event map")
map_df = (df_f[[EVENT_ID_COL, DATE_COL, COUNTRY_COL, ADMIN1_COL, EVTYPE_COL, SUBEV_COL, LAT_COL, LON_COL, TEXT_COL]]
          .dropna(subset=[LAT_COL, LON_COL]))
if not map_df.empty:
    fig = px.scatter_mapbox(
        map_df, lat=LAT_COL, lon=LON_COL,
        hover_name=EVENT_ID_COL,
        hover_data=[DATE_COL, COUNTRY_COL, ADMIN1_COL, EVTYPE_COL, SUBEV_COL],
        color=EVTYPE_COL, zoom=3, height=500
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No coordinates to plot.")

# --------------------------
# Distributions
# --------------------------
st.subheader("Top actions & micro-types")
left, right = st.columns(2)
with left:
    st.bar_chart(micro["action_lemma"].value_counts().head(20))
with right:
    st.bar_chart(micro["micro_type"].value_counts().head(20))

# --------------------------
# Topic sketch + inter-linkages (same as v1, trimmed)
# --------------------------
st.subheader("Event inter-linkages (same year & admin1)")
if not df_f.empty and df_f[ADMIN1_COL].notna().any():
    link_year = st.selectbox("Year", sorted(df_f["year"].dropna().unique()))
    link_admin1 = st.selectbox("Admin1", sorted(df_f[ADMIN1_COL].dropna().unique()))
    sub = df_f[(df_f["year"]==link_year) & (df_f[ADMIN1_COL]==link_admin1)].dropna(subset=[TEXT_COL, LAT_COL, LON_COL, DATE_COL])
    if len(sub) > 1:
        tfidf2 = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
        X2 = tfidf2.fit_transform(sub["text"].tolist())
        cos = cosine_similarity(X2)
        G = nx.Graph()
        ids = sub[EVENT_ID_COL].tolist()
        dates = sub[DATE_COL].tolist()
        lats  = sub[LAT_COL].tolist()
        lons  = sub[LON_COL].tolist()

        for r in sub.itertuples():
            G.add_node(getattr(r, EVENT_ID_COL), date=getattr(r, DATE_COL), admin1=getattr(r, ADMIN1_COL))

        K = 8
        for i in range(len(sub)):
            nbr = np.argsort(cos[i])[::-1][1:K+1]
            for j in nbr:
                dt = abs((dates[i] - dates[j]).days)
                dd = haversine(lats[i], lons[i], lats[j], lons[j])
                time_k = math.exp(-dt/14) if pd.notna(dt) else 0.0
                space_k = math.exp(-((dd or 0)/50)) if dd==dd else 0.0
                w = 0.5*float(cos[i,j]) + 0.25*time_k + 0.25*space_k
                if w >= 0.45:
                    G.add_edge(ids[i], ids[j], weight=round(w,3))
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        st.write(f"Components: {len(comps)} — sizes (top 10): {[len(c) for c in comps[:10]]}")
        if comps:
            largest = list(comps[0])
            tab = sub[sub[EVENT_ID_COL].isin(largest)][[EVENT_ID_COL, DATE_COL, ADMIN1_COL, EVTYPE_COL, SUBEV_COL, TEXT_COL]]
            st.dataframe(tab.sort_values(DATE_COL).head(100), use_container_width=True)
    else:
        st.info("Not enough events in this partition.")
else:
    st.info("No Admin1 values available.")

# --------------------------
# Downloads
# --------------------------
st.subheader("Downloads")
st.download_button("Micro-event table (CSV)", micro.to_csv(index=False).encode("utf-8"),
                   file_name="micro_events_v2.csv", mime="text/csv")
st.download_button("Partition summary (CSV)", agg.to_csv(index=False).encode("utf-8"),
                   file_name="partition_summary_v2.csv", mime="text/csv")

st.caption("Tips: Extend VERB_CANON & TARGETS for your theater (e.g., FARDC, MONUSCO, M23, specific roads/bridges).")

# ACLED Conflict Analytics

This repository hosts a Streamlit application for exploring conflict event data from the Armed Conflict Location & Event Data Project (ACLED). The app ingests the provided Excel or CSV export, harmonises its schema, and surfaces interactive analytics that combine geospatial summaries, semantic search, unsupervised clustering, and actor network exploration.

## Key capabilities

- **Robust data ingestion** – validates the ACLED export, coerces typed columns, and caches the cleaned dataset for faster reloads.
- **Global filtering controls** – constrain the working subset by time, geography, actors, and free-text search terms with immediate feedback.
- **Overview dashboards** – view high-level metrics, temporal trends, fatality summaries, and faceted bar charts.
- **Semantic event search** – retrieve notes similar to a natural-language query using TF–IDF embeddings and cosine similarity, followed by strict lexical filtering to limit false positives.
- **Context-aware clustering** – run configurable K-Means clustering on numeric, categorical, and contextual text features with silhouette quality diagnostics and automatic theme summaries.
- **Actor–category network** – analyse bipartite connections between primary actors and their ACLED categories with degree-weighted node sizing and square-root-normalised edge widths.
- **Detailed event table** – exportable tabular view of the filtered dataset with contextual tooltips.

## Repository structure

```
ACLED_app/
├── ACLED_analytics.py      # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project overview (this file)
└── docs/
    └── methodology.tex     # Mathematical description of the analytics pipeline
```

## Getting started

1. **Install Python dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Provide the ACLED dataset**
   Place the Excel export as `ACLED 2016-2025.xlsx` in the repository root. The application will automatically fall back to any CSV files in the same directory if the Excel file is absent.
3. **Run the Streamlit app**
   ```bash
   streamlit run ACLED_analytics.py
   ```
4. **Interact with the analytics**
   Use the left-hand sidebar to filter events. Explore the overview, semantic search, clustering, network, and data tabs to interrogate patterns, actor interactions, and thematic clusters.

## Development notes

- The application caches the cleaned dataset (`acled_dataset.pkl`) to avoid repeated parsing of the source file. Delete the cache if you refresh the ACLED export.
- Most transformations are implemented in `ACLED_analytics.py`. See `docs/methodology.tex` for a rigorous description of the statistical and algorithmic techniques applied to each component.
- Run `python -m compileall ACLED_analytics.py` to perform a lightweight syntax check before committing changes.

## License

No license information is provided. Please consult the data provider's terms of use before distributing ACLED data or derivative analyses.

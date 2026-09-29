# YSocial Analytics Dashboard — Standalone Setup

This dashboard reads a single YSocial `database_server.db` file. It does
**not** require Ollama, YSocial itself, or any conda environment — just
Python and six packages (streamlit, pandas, networkx, matplotlib, scipy,
plotly).

## Files

Three Python files must sit in the same folder:

- `streamlit_dashboard.py` — the app
- `ysocial_network.py` — builds both network layers (follow layer from the
  event log with unfollows applied, Admin account excluded, interaction
  layer from likes, dislikes and replies)
- `network_layout.py` — community layout and colors

The thesis analysis scripts (`phase1_validate.py`, `phase2_analysis.py`,
`phase3_temporal.py`) use the same two modules, so the dashboard shows the
same numbers as their outputs.

## Setup (once)

```bash
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment.)

For the dark theme, put `config.toml` into a `.streamlit` folder next to
the script:

```bash
mkdir -p .streamlit && cp config.toml .streamlit/config.toml
```

## Run

```bash
streamlit run streamlit_dashboard.py
```

Your browser should open automatically. If not, open the URL shown in the
terminal (usually `http://localhost:8501`).

## Loading data

The sidebar accepts a database in three ways (first match wins):

1. **Upload** — drag and drop any YSocial `database_server.db` onto the
   upload box. Works identically on Mac, Windows, or Linux.
2. **Local path** — type the path to a database, e.g.
   `~/YSocial/y_web/experiments/<id>/database_server.db`. The **Compare**
   tab then compares this run with every run stored next to it: each
   sibling folder that holds a `database_server.db` (YSocial's experiments
   folder, or any folder you organise the same way, such as
   `data/<run>/database_server.db`).
3. **Next to the script** — if a file named `database_server.db` is in the
   same folder as `streamlit_dashboard.py`, it loads automatically. This is
   the setup for a cloud deployment (e.g. Streamlit Community Cloud).

The dashboard only reads the file; it never modifies it.

## What is interactive and what is not

- All analytical charts (Interactions, Groups, Topics, Text, Comments,
  Network × Content, Compare) are Plotly charts: hover for exact values,
  drag to zoom, double-click to reset.
- The **Network** tab is static on purpose: it is about a readable network
  layout, not an exploration tool. Each community has its own region and
  color (Kamada-Kawai inside a region), ties between communities are faint,
  or bundled into one line per pair of communities in a dense follow
  network, and a community summary graph shows which communities are most
  tied to each other. The layout is the same as in the thesis figures.

## Which database to try

Any `database_server.db` produced by a YSocial experiment works. For the
most informative view, use one from a run with a seeded starting network
(for example Watts-Strogatz with the Preferential Attachment recommender):
it has a populated follow network, so both layers and their comparison can
be shown. Runs without a seeded network have an almost empty follow layer.

The Network × Content tab is a secondary analysis (S1–S3: centrality and
sentiment, bridge vs. core tone, sentiment across communities). It is not
part of the thesis research questions RQ1a–RQ4.

## Performance

Networks, communities, centrality and the network images are cached per
database for 30 seconds. On a 1,500-agent test database the first load
takes about 9 seconds and later page loads about 2 seconds.

## Column-name note

Table and column names (`follow.user_id` = follower, `follow.follower_id` =
followee, `post.comment_to`, `reactions.type`, the `Admin` account…) are
defined once, at the top of `ysocial_network.py`, and were checked against
the YSocial source code. If you point the dashboard at a database from a
different YSocial version, check them there; the dashboard and the analysis
scripts both read them from that one place.

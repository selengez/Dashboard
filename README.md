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
# YSocial Analytics Dashboard

This dashboard was developed as part of my thesis on network analytics for LLM-driven social media simulations using YSocial.

The dashboard reads a YSocial `database_server.db` file and presents the main activity, network and content measures from a simulation run.

The current thesis focuses mainly on two network layers:

- **Designed follow layer** — the follow network given to the agents at the start of the simulation.
- **Emergent interaction layer** — the network created through likes, dislikes and replies during the simulation.

The dashboard uses the same network-building rules as the thesis analysis scripts, so the figures and summary values stay consistent across the project.

## Live dashboard

The deployed dashboard is available here:

**https://ysocial-thesis-dashboard.streamlit.app**

## Main files

The main Python files are:

- `streamlit_dashboard.py` — Streamlit application
- `ysocial_network.py` — builds the follow and interaction layers
- `network_layout.py` — community-based network layout used in the dashboard and thesis figures

The analysis scripts (`phase1_validate.py`, `phase2_analysis.py`, and `phase3_temporal.py`) use the same shared network code.

## Network definitions

The dashboard follows the corrected network definitions used in the thesis.

### Follow layer

The YSocial `follow` table is an event log rather than a simple edge list.

The dashboard therefore:

- keeps only the latest action for each follower–followee pair;
- removes ties whose latest action is `unfollow`;
- excludes the YSocial `Admin` account from the simulated-agent node set.

### Interaction layer

The interaction layer includes:

- likes;
- dislikes;
- replies.

An edge goes from the acting agent to the author of the post or parent post.

Repeated interactions between the same two agents increase the edge weight.

## Dashboard sections

### Interactions

Shows posts and reactions over simulation rounds.

The chart is interactive and supports hover, zoom and pan.

### Network

The Network section is intentionally **static**.

This follows the supervisor's feedback to focus on network visualization and layout rather than interaction controls.

The layout is designed to make the network easier to read:

- each Louvain community has its own region;
- each community has its own color;
- agents are arranged with Kamada-Kawai inside their community region;
- node size reflects degree centrality;
- only the most central agents are labelled;
- ties inside communities use the community color;
- ties between communities are lighter;
- in dense follow networks, links between communities are bundled;
- a community-level summary graph shows the strongest connections between communities.

The same layout logic is also used in the thesis figures.

### Groups

Shows the sizes of the interaction-layer communities detected with Louvain.

The chart is interactive.

### Topics

Shows the distribution of post topics.

The chart is interactive.

### Text

Shows sentiment and toxicity distributions when those annotations are available.

The charts are interactive.

### Comments

Shows the posts receiving the most replies.

The chart is interactive.

### Network × Content

This is a **secondary analysis**, separate from the main thesis research questions.

It currently includes:

- **S1:** centrality and sentiment;
- **S2:** bridge vs. core tone;
- **S3:** sentiment across communities;
- follow vs. interaction centrality as an additional node-position comparison.

These charts are interactive.

### Compare

The Compare tab places multiple simulation runs side by side.

It is intended mainly for the pilot and later experimental runs.

Measures include:

- number of agents;
- seed followees per agent;
- follow ties;
- active-agent share;
- interaction ties;
- interaction-layer modularity;
- interaction volume on follow ties;
- followed share of the timeline.

## Interactive vs. static views

The dashboard deliberately uses two different visualization styles:

- **Analytical charts:** interactive Plotly figures with hover, zoom and pan.
- **Network visualization:** static Matplotlib figures focused on layout and readability.

This separation directly follows the supervisor feedback on the dashboard.

## Running locally

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then start the dashboard:

```bash
streamlit run streamlit_dashboard.py
```

The browser should open automatically. If it does not, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Loading data

The dashboard can load a database in three ways.

### 1. Upload

Upload any YSocial `database_server.db` file from the sidebar.

### 2. Local path

Enter the path to a database, for example:

```text
~/YSocial/y_web/experiments/<experiment_id>/database_server.db
```

When a local experiment database is used, the **Compare** tab can also find sibling experiment folders stored next to it.

### 3. Database next to the script

If a file named `database_server.db` is placed in the same folder as `streamlit_dashboard.py`, it loads automatically.

This is the setup used for cloud deployment.

The dashboard only reads the database. It does not modify it.

## Recommended run

Any YSocial `database_server.db` file can be opened.

For the most informative network view, use a run with a seeded starting network, for example a Watts-Strogatz network with the Preferential Attachment friendship recommender. These runs contain a populated follow layer and allow the designed and emergent layers to be compared directly.

Runs without a seeded network may have an empty or nearly empty follow layer.

## Current thesis run

For the current main pilot (`Thesis_scaled_v1`), the corrected network construction gives:

- 300 simulated agents;
- 1,203 final follow ties;
- 87 interaction-active agents;
- 191 interaction ties;
- 275 total interactions.

These values match the current thesis analysis.

## Performance

Networks, communities, centrality measures and network figures are cached by Streamlit.

This keeps repeated page loads fast while still allowing the dashboard to refresh when the database changes.

## Project structure

A typical local setup is:

```text
Dashboard/
├── streamlit_dashboard.py
├── ysocial_network.py
├── network_layout.py
├── requirements.txt
├── database_server.db
└── .streamlit/
    └── config.toml
```

## Notes on YSocial schema

The dashboard uses the YSocial schema checked against the current project source code.

Examples include:

- `follow.user_id` = follower;
- `follow.follower_id` = followed agent;
- `post.comment_to` = parent post for replies;
- `reactions.type` = reaction type;
- `Admin` = platform account, not a simulated agent.

If the dashboard is used with a different YSocial version, these definitions should be checked again.

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

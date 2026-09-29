"""
YSocial Analytics Dashboard
===========================

Streamlit dashboard for the YSocial thesis project. Runs locally against a
YSocial experiment folder, or in the cloud with a database file placed next
to this script.

Design choices:
- Streamlit is the application framework (layout, tabs, widgets).
- Analytical charts are drawn with Plotly and rendered through
  st.plotly_chart, so they support hover, zoom and pan.
- The network visualization section is static on purpose (supervisor's
  brief): it is about a readable layout, not an exploration tool. Figures
  are drawn with matplotlib using network_layout.py, the same layout as the
  thesis figures.
- Networks are built with ysocial_network.py and laid out with
  network_layout.py, the same modules the thesis analysis scripts use, so
  every number here matches the Phase 1-3 outputs. Both files must sit
  next to this script.

RUN:
    pip install -r requirements.txt
    streamlit run streamlit_dashboard.py

Data source priority (sidebar):
    1. an uploaded database_server.db
    2. a local path typed in the sidebar (e.g. a YSocial experiment folder)
    3. database_server.db placed next to this script (cloud deployment)
"""

import hashlib
import inspect
import json
import sqlite3
import tempfile
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.patches import Patch

import plotly.graph_objects as go
import streamlit as st

# Shared with the thesis analysis scripts (phase1_validate.py, phase2_analysis.py,
# phase3_temporal.py): one definition of both network layers, one layout.
import ysocial_network as yn
import network_layout as nl


# ============================================================
# Verified YSocial schema
# ============================================================
POST_ID_COL = "id"
POST_AUTHOR_COL = "user_id"
COMMENT_TO_COL = "comment_to"
ROUND_COL = "round"
REACTION_ACTOR_COL = "user_id"
REACTION_TARGET_COL = "post_id"
FOLLOW_SOURCE_COL = "user_id"
FOLLOW_TARGET_COL = "follower_id"
SENTIMENT_SCORE_COL = "compound"
TOXICITY_SCORE_COL = "toxicity"

# Design tokens (dark UI). The same values are set as the Streamlit theme in
# .streamlit/config.toml, so Streamlit's own widgets, the Plotly charts and
# the matplotlib network figures share one palette.
BG = "#0F1117"          # page background
SURFACE = "#151821"     # cards, figure panels
ELEVATED = "#1B1F2A"    # tooltips, hover labels
BORDER = "#2A2F3A"      # 1px borders, grid lines
TEXT = "#F1F3F7"        # primary text
TEXT2 = "#9BA3B0"       # secondary text
MUTED = "#6F7785"       # metadata, axis ticks
ACCENT = "#8B7CF6"      # the one accent color
POSITIVE = "#3FBF7F"
NEGATIVE = "#E45F65"

# Older names kept so the rest of the script reads the same.
NEUTRAL, SUCCESS, DANGER = MUTED, POSITIVE, NEGATIVE
CARD_BG, GRID = SURFACE, BORDER
FONT_FAMILY = '"Source Sans", "Source Sans Pro", sans-serif'  # Streamlit's own UI font

st.set_page_config(
    page_title="YSocial Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT2,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": CARD_BG,
    "axes.facecolor": CARD_BG,
    "savefig.facecolor": CARD_BG,
})


# ============================================================
# Streamlit version compatibility
# ------------------------------------------------------------
# Recent Streamlit versions replaced `use_container_width=True` with
# `width="stretch"` and warn about the old argument. Older versions only
# know the old argument. Pick whichever the installed version supports.
# ============================================================
def _full_width_kwargs(func):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    width_param = params.get("width")
    if width_param is not None and isinstance(width_param.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def _supported(func, **kwargs):
    """Keep only the keyword arguments the installed Streamlit knows, so
    newer styling options degrade quietly on older versions."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


PLOTLY_WIDTH = _full_width_kwargs(st.plotly_chart)
PYPLOT_WIDTH = _full_width_kwargs(st.pyplot)
TABLE_WIDTH = _full_width_kwargs(st.dataframe)
IMAGE_WIDTH = _full_width_kwargs(st.image)


# st.metric normally renders a delta as a green/red arrow ("better/worse").
# A p-value is not a direction, so show it neutrally: no color, and no arrow
# where the installed Streamlit supports hiding it.
NEUTRAL_DELTA = {"delta_color": "off"}
if "delta_arrow" in inspect.signature(st.metric).parameters:
    NEUTRAL_DELTA["delta_arrow"] = "off"


def p_label(p):
    """Readable significance label for a p-value (alpha = 0.05)."""
    return f"p = {p:.3f} · {'significant' if p < 0.05 else 'n.s.'}"


# ============================================================
# Plotly helpers — interactive analytical charts
# ============================================================
def style_plotly(fig, height=390, x_title=None, y_title=None):
    axis = dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=False,
                tickfont=dict(color=MUTED, size=12), title_font=dict(color=TEXT2, size=13))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT2, family=FONT_FAMILY, size=13),
        hoverlabel=dict(bgcolor=ELEVATED, bordercolor=BORDER, font_color=TEXT,
                        font_family=FONT_FAMILY),
        # Quiet toolbar: shown on hover, in the muted text color.
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=ACCENT),
        xaxis=dict(title=x_title, **axis),
        yaxis=dict(title=y_title, **axis),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def interactive_line_fig(rounds, posts_values, reaction_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds,
        y=posts_values,
        mode="lines",
        name="Posts",
        line=dict(color=ACCENT, width=2),
        hovertemplate="Round %{x}<br>Posts: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rounds,
        y=reaction_values,
        mode="lines",
        name="Reactions",
        line=dict(color=TEXT2, width=2, dash="dot"),
        hovertemplate="Round %{x}<br>Reactions: %{y}<extra></extra>",
    ))
    fig.update_layout(hovermode="x unified", dragmode="zoom")
    return style_plotly(fig, height=420, x_title="Simulation round", y_title="Count")


def interactive_bar_fig(labels, values, horizontal=False, color=ACCENT, height=390):
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color),
            hovertemplate="%{y}<br>Count: %{x}<extra></extra>",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return style_plotly(fig, height=height, x_title="Count", y_title=None)

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=color),
        hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
    ))
    return style_plotly(fig, height=height, x_title=None, y_title="Count")


def interactive_histogram_fig(values, start, end, size, x_title, color=ACCENT):
    fig = go.Figure(go.Histogram(
        x=values,
        xbins=dict(start=start, end=end, size=size),
        marker=dict(color=color),
        hovertemplate=f"{x_title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ))
    return style_plotly(fig, height=360, x_title=x_title, y_title="Posts")


def interactive_sparkline(values, color=ACCENT):
    x = list(range(len(values)))
    fig = go.Figure(go.Scatter(
        x=x,
        y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor="rgba(139,124,246,0.10)",
        hovertemplate="Value: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=48,
        margin=dict(l=0, r=0, t=2, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ============================================================
# Static network visualization — layout-focused by design
# ============================================================
def _community_map(communities):
    return {node: idx for idx, community in enumerate(communities) for node in community}


# Community colors — one list used by EVERY view (network views, Groups bar
# chart, S1 scatter, S3 box plot), so C3 is the same color everywhere.
# Communities are sorted by size, so C0 is always the largest. Each community
# has its own color (network_layout.distinct_color); with many communities
# some colors are close, so the network views also name the community on
# hover and let the legend hide or isolate communities.
def community_color(cid):
    """Every community gets its own color, the same in every view."""
    return nl.distinct_color(cid, "dark")


def _label_ink(hex_color):
    """Dark or light text, whichever reads better on this fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return BG if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.35 else TEXT


def _rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


BUNDLE_ABOVE = 300   # ties between communities; above this, bundle them


def _halo():
    return [patheffects.withStroke(linewidth=2.6, foreground=SURFACE)]


def static_network_fig(G, communities, subtitle, top_labels=5, figsize=(9.6, 7.2)):
    """Layout-focused network figure, static by design (no hover or zoom):
    the network visualization section is about the layout itself.

    - layout: network_layout.community_grouped_layout (each community in its
      own region, Kamada-Kawai inside a community), the same as the thesis
      figures
    - every community has its own color; regions are only a 7% wash
    - ties inside a community in its color; ties between communities faint
      and gray, or, in a dense layer, bundled into one line per pair of
      communities (width = number of ties, strongest pairs only)
    - node size = degree centrality; only the most central agents labelled,
      with a halo so labels stay readable over edges
    - legend in its own column on the right"""
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.01, 0.01, 0.74, 0.92])
    ax.set_facecolor(SURFACE)
    if G.number_of_edges() == 0 or not communities:
        ax.text(0.5, 0.5, "No network edges available", ha="center", va="center",
                color=TEXT2, transform=ax.transAxes)
        ax.axis("off")
        return fig

    pos, centers, radii = nl.community_grouped_layout(G, communities)
    lookup = _community_map(communities)
    k = len(communities)
    centers, radii = np.asarray(centers)[:k], np.asarray(radii)[:k]

    for cid in range(k):
        ax.add_patch(mpl.patches.Circle(centers[cid], radii[cid] * 1.04, zorder=0,
                                        facecolor=community_color(cid), alpha=0.07,
                                        edgecolor="none"))

    inter = [(u, v) for u, v in G.edges() if lookup.get(u) != lookup.get(v)]
    dense = G.number_of_edges() > 600
    if len(inter) > BUNDLE_ABOVE:
        links = inter_community_links(G, communities)[:max(3, int(1.5 * k))]
        top = links[0][1] if links else 1
        for (a, b), count in sorted(links, key=lambda kv: kv[1]):
            ax.plot([centers[a][0], centers[b][0]], [centers[a][1], centers[b][1]],
                    color=TEXT2, alpha=0.14 + 0.26 * count / top, linewidth=0.8 + 5.5 * count / top,
                    solid_capstyle="round", zorder=1)
    elif inter:
        nx.draw_networkx_edges(G, pos, edgelist=inter, ax=ax, width=0.6, alpha=0.16,
                               edge_color=TEXT2, arrows=False)
    weights = {(u, v): d.get("weight", 1) for u, v, d in G.edges(data=True)}
    max_w = max(weights.values()) if weights else 1
    for cid in range(k):
        intra = [(u, v) for u, v in G.edges() if lookup.get(u) == cid and lookup.get(v) == cid]
        if intra:
            nx.draw_networkx_edges(
                G, pos, edgelist=intra, ax=ax, arrows=False, edge_color=community_color(cid),
                alpha=0.22 if dense else 0.38,
                width=[(0.5 if dense else 0.6) + 1.0 * weights[e] / max_w for e in intra])

    centrality = nx.degree_centrality(G)
    max_c = max(max(centrality.values()), 1e-9)
    nodes = list(G.nodes())
    xy = np.array([pos[n] for n in nodes])
    lo = np.minimum(xy.min(axis=0), (centers - radii[:, None] * 1.04).min(axis=0)) - 0.03
    hi = np.maximum(xy.max(axis=0), (centers + radii[:, None] * 1.04).max(axis=0)) + 0.03
    # Node size from the spacing the layout actually produced: the largest
    # node is about as wide as the typical gap to a neighbour, so nodes in a
    # dense community don't pile on top of each other.
    from scipy.spatial import cKDTree
    gap = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])) if len(xy) > 1 else 0.1
    pt_per_unit = 0.74 * figsize[0] * 72 / max(hi[0] - lo[0], hi[1] - lo[1])
    max_area = min((1.15 * gap * pt_per_unit) ** 2, 520)
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, ax=ax, alpha=0.95, linewidths=0.7, edgecolors=SURFACE,
        node_color=[community_color(lookup.get(n, 0)) for n in nodes],
        node_size=[max_area * (0.22 + 0.78 * (centrality[n] / max_c) ** 0.8) for n in nodes])
    for n in sorted(centrality, key=centrality.get, reverse=True)[:top_labels]:
        ax.annotate(str(n), pos[n], xytext=(5, 5), textcoords="offset points", fontsize=9,
                    color=TEXT, weight="semibold", zorder=6, path_effects=_halo())

    fig.text(0.015, 0.975, subtitle, ha="left", va="top", fontsize=10, color=TEXT2)
    handles = [Patch(facecolor=community_color(i), edgecolor="none",
                     label=f"C{i} · {len(c)} agents") for i, c in enumerate(communities)]
    if len(inter) > BUNDLE_ABOVE:
        handles.append(mpl.lines.Line2D([], [], color=TEXT2, alpha=0.4, linewidth=3,
                                        label="links between communities"))
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.765, 0.93),
                     frameon=False, fontsize=9 if k <= 16 else 8, labelcolor=TEXT2,
                     handlelength=0.9, handleheight=0.9, labelspacing=0.55,
                     ncol=1 if k <= 22 else 2, title="Communities", title_fontsize=9.5,
                     alignment="left")
    leg.get_title().set_color(TEXT)

    # Fit the view to what is drawn (nodes and regions), equal aspect.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    return fig


def static_community_summary_fig(G, communities, figsize=(7.2, 5.4)):
    """One circle per community (area ~ number of agents, labelled) and a line
    for each of the strongest community pairs (width ~ number of ties)."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    if not communities or G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No communities to summarize", ha="center", va="center",
                color=TEXT2, transform=ax.transAxes)
        ax.axis("off")
        return fig
    _, centers, radii = nl.community_grouped_layout(G, communities)
    k = len(communities)
    centers = np.asarray(centers)[:k]
    links = inter_community_links(G, communities)[:max(3, int(1.5 * k))]
    top = links[0][1] if links else 1
    for (a, b), count in sorted(links, key=lambda kv: kv[1]):
        ax.plot([centers[a][0], centers[b][0]], [centers[a][1], centers[b][1]],
                color=TEXT2, alpha=0.18 + 0.45 * count / top, linewidth=0.8 + 6 * count / top,
                solid_capstyle="round", zorder=1)
    sizes = np.array([len(c) for c in communities], dtype=float)
    ax.scatter(centers[:, 0], centers[:, 1], s=260 + 1500 * sizes / sizes.max(),
               c=[community_color(i) for i in range(k)], edgecolors=SURFACE, linewidths=1.5,
               zorder=3)
    for i, (x, y) in enumerate(centers):
        ax.text(x, y, f"C{i}\n{int(sizes[i])}", ha="center", va="center", fontsize=8.5,
                color=_label_ink(community_color(i)), weight="semibold", zorder=4)
    pad = 0.2
    ax.set_xlim(centers[:, 0].min() - pad, centers[:, 0].max() + pad)
    ax.set_ylim(centers[:, 1].min() - pad, centers[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig


def community_share_stats(G, communities, weight=None):
    """Share of ties that stay inside a community, plus Louvain modularity.
    Computed on the undirected projection (weights of both directions
    summed), exactly as in phase2_analysis.py: weighted for the interaction
    layer, unweighted for the follow layer."""
    lookup = _community_map(communities)
    H = yn.undirected_projection(G)
    if H.number_of_edges() == 0 or not communities:
        return None, None
    inside = sum(1 for u, v in H.edges() if lookup.get(u) == lookup.get(v))
    modularity = nx.algorithms.community.modularity(H, communities, weight=weight)
    return inside / H.number_of_edges(), modularity


def inter_community_links(G, communities):
    """Number of (undirected) ties between each pair of communities,
    strongest first: [((a, b), count), ...]."""
    lookup = _community_map(communities)
    between = {}
    for u, v in G.to_undirected().edges():
        a, b = lookup.get(u), lookup.get(v)
        if a is not None and b is not None and a != b:
            key = (min(a, b), max(a, b))
            between[key] = between.get(key, 0) + 1
    return sorted(between.items(), key=lambda kv: kv[1], reverse=True)


def half_split_pct_change(series):
    vals = list(series)
    if len(vals) < 4:
        return None
    mid = len(vals) // 2
    first, second = sum(vals[:mid]), sum(vals[mid:])
    if first == 0:
        return None
    return (second - first) / first * 100


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=30)
def load_data(db_path):
    """Network tables through ysocial_network.load_tables (read-only), plus
    the content tables the dashboard shows."""
    t = yn.load_tables(db_path)
    data = {
        "posts": t["post"],
        "reactions": t["reactions"],
        "follow": t["follow"],
        "users": t["user_mgmt"],
    }
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for table in ["post_sentiment", "post_toxicity", "post_topics"]:
        try:
            data[table] = pd.read_sql(f"SELECT * FROM {table};", conn)
        except Exception:
            data[table] = pd.DataFrame()
    conn.close()
    return data


def find_runs(current_db_path):
    """Runs to compare: every sibling folder of the loaded run's folder that
    holds a database_server.db. Works for YSocial's own experiments folder
    (…/experiments/<id>/database_server.db) and for any folder organised
    the same way (e.g. data/<run>/database_server.db)."""
    current = os.path.abspath(current_db_path)
    parent = os.path.dirname(os.path.dirname(current))
    runs = []
    for folder in sorted(glob.glob(os.path.join(parent, "*"))):
        db_file = os.path.join(folder, "database_server.db")
        if not os.path.isfile(db_file):
            continue
        name = os.path.basename(folder)
        config_path = os.path.join(folder, "config_server.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    name = json.load(f).get("name", name) or name
            except Exception:
                pass
        runs.append({"name": name, "path": db_file, "mtime": os.path.getmtime(db_file),
                     "current": os.path.abspath(db_file) == current})
    if not any(r["current"] for r in runs):
        runs.append({"name": os.path.basename(os.path.dirname(current)), "path": current,
                     "mtime": os.path.getmtime(current), "current": True})
    return parent, runs


@st.cache_data(show_spinner=False)
def run_metrics(db_file, mtime):
    """The comparison numbers for one run, computed with the same shared rules
    as the thesis scripts. Cached per file and modification time, so a run is
    recomputed only when its database changes."""
    t = yn.load_tables(db_file)
    posts, reactions, follow = t["post"], t["reactions"], t["follow"]
    nodes = yn.node_universe(t["user_mgmt"])
    cal = yn.round_calendar(t["rounds"])
    rounds = len(cal) if cal is not None else (int(posts[yn.POST_ROUND].max()) + 1 if len(posts) else 0)

    G_int = yn.interaction_layer(yn.interaction_events(posts, reactions), nodes)
    G_fol = yn.follow_layer(follow, nodes)
    active = yn.active_agents(G_int)
    A = G_int.subgraph(active)
    W = sum(d["weight"] for _, _, d in A.edges(data=True))
    on_follow = sum(d["weight"] for u, v, d in A.edges(data=True) if G_fol.has_edge(u, v))
    H = yn.undirected_projection(A)
    comms = nl.louvain(H, weight="weight") if H.number_of_edges() else []
    seed = yn.follow_state(follow, before_round=yn.SEED_ROUND + 1)
    days = rounds / 24 if rounds else float("nan")
    n = len(nodes) or 1
    return {
        "agents": len(nodes),
        "simulated hours": rounds,
        "seed followees per agent": len(seed) / n,
        "follow ties": G_fol.number_of_edges(),
        "posts": len(posts),
        "reactions": len(reactions),
        "active agents": len(active),
        "active share": len(active) / n,
        "interaction ties": A.number_of_edges(),
        "interactions / agent / day": W / n / days if days else float("nan"),
        "modularity": nx.algorithms.community.modularity(H, comms, weight="weight") if comms else float("nan"),
        "volume on follow ties": on_follow / W if W else float("nan"),
        "followed share of timeline": yn.followed_timeline_share(t["recommendations"], posts, follow),
    }


# Both layers come from ysocial_network.py (see its docstring for the rules):
# - node set: every simulated agent; YSocial's Admin account is excluded
# - follow layer: rebuilt from the follow event log, keeping the latest
#   action per (follower, followee) pair, so unfollowed ties are dropped
# - interaction layer: actor -> author for every like, dislike and reply,
#   weighted by the number of interactions, self-ties dropped


# ============================================================
# Cached derived results
# ------------------------------------------------------------
# Everything expensive is computed once per database and reused on every
# rerun (widget change, page refresh, another viewer). The cache key is the
# database path, and the 30-second TTL matches load_data(), so a running
# simulation is still picked up within 30 seconds.
# ============================================================
@st.cache_data(ttl=30, show_spinner="Building networks and communities…")
def compute_networks(db_path):
    data = load_data(db_path)
    nodes = yn.node_universe(data["users"])
    events = yn.interaction_events(data["posts"], data["reactions"])
    G_interaction = yn.interaction_layer(events, nodes)
    G_follow = yn.follow_layer(data["follow"], nodes)

    active = yn.active_agents(G_interaction)
    isolated = [n for n in G_interaction.nodes() if G_interaction.degree(n) == 0]
    G_active = G_interaction.subgraph(active).copy()
    # Same Louvain call and seed as phase2_analysis.py: weighted for the
    # interaction layer, unweighted for the follow layer.
    communities = (
        nl.louvain(yn.undirected_projection(G_active), weight="weight")
        if G_active.number_of_edges() > 0 else []
    )

    G_follow_active = G_follow.subgraph(
        [n for n in G_follow.nodes() if G_follow.degree(n) > 0]).copy()
    follow_communities = (
        nl.louvain(yn.undirected_projection(G_follow_active), weight=None)
        if G_follow_active.number_of_edges() > 0 else []
    )
    return {
        "G_interaction": G_interaction, "G_follow": G_follow,
        "G_active": G_active, "G_follow_active": G_follow_active,
        "isolated": isolated, "communities": communities,
        "follow_communities": follow_communities,
    }


@st.cache_data(ttl=30, show_spinner="Computing centrality…")
def compute_centrality(db_path):
    nets = compute_networks(db_path)
    G_active, G_follow = nets["G_active"], nets["G_follow"]
    return {
        "degree": nx.degree_centrality(G_active),
        "betweenness": nx.betweenness_centrality(G_active),
        "follow_degree": nx.degree_centrality(G_follow) if G_follow.number_of_edges() > 0 else {},
    }


@st.cache_data(ttl=30, show_spinner="Laying out the network…")
def network_png(db_path, which, dpi=160):
    """Render one network figure to PNG bytes. Cached per database, so the
    layout (the slowest step at 1,000+ agents) runs once, not on every rerun."""
    nets = compute_networks(db_path)
    if which == "interaction":
        G, comms = nets["G_active"], nets["communities"]
        fig = static_network_fig(G, comms, f"{G.number_of_nodes()} active agents · "
                                 f"{G.number_of_edges()} directed ties · {len(comms)} communities")
    elif which == "follow":
        G, comms = nets["G_follow_active"], nets["follow_communities"]
        fig = static_network_fig(G, comms, f"{G.number_of_nodes()} agents · "
                                 f"{G.number_of_edges():,} directed ties · {len(comms)} communities")
    elif which == "follow_summary":
        fig = static_community_summary_fig(nets["G_follow_active"], nets["follow_communities"])
    else:
        raise ValueError(which)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


# ============================================================
# Sidebar / cloud-friendly data selection
# ============================================================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB = APP_DIR / "database_server.db"

st.sidebar.markdown("### YSocial Analytics")
st.sidebar.caption("Network × content dashboard")
st.sidebar.divider()
st.sidebar.markdown("**Data source**")

uploaded_file = st.sidebar.file_uploader(
    "Upload a YSocial database",
    type=["db"],
    help="Optional: upload a database_server.db file for this browser session.",
)
manual_path = st.sidebar.text_input(
    "…or local database path",
    value=str(DEFAULT_DB) if DEFAULT_DB.is_file() else "",
    placeholder="~/YSocial/y_web/experiments/<id>/database_server.db",
    help="Point to a database inside a YSocial experiment folder to also enable the Compare tab.",
)
manual_path = os.path.expanduser(manual_path.strip())


def _persist_upload(file):
    """Write the upload to a path derived from its content hash, so the same
    file always maps to the same path and cached results are reused across
    reruns (a fresh random temp file each rerun would defeat the cache)."""
    content = file.getvalue()
    digest = hashlib.md5(content).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"ysocial_upload_{digest}.db")
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(content)
    return path


if uploaded_file is not None:
    db_path = _persist_upload(uploaded_file)
    single_file_mode = True
    st.sidebar.success("Uploaded database loaded")
elif manual_path and os.path.isfile(manual_path):
    db_path = manual_path
    single_file_mode = False
    st.sidebar.success("Local database loaded")
else:
    if manual_path:
        st.sidebar.error("No file found at that path.")
    st.markdown("## YSocial Analytics")
    st.warning("No simulation database is loaded yet.")
    st.write(
        "Upload a YSocial `database_server.db` file in the sidebar, or type the path "
        "to one on this machine. The dashboard only reads the file; it never modifies it."
    )
    st.stop()

try:
    data = load_data(db_path)
except Exception as e:
    st.error(f"Could not open this database: {e}")
    st.stop()


# ============================================================
# Network preparation (cached, see compute_networks)
# ============================================================
posts = data["posts"]
reactions = data["reactions"]
follow = data["follow"]
users = data["users"]

nets = compute_networks(db_path)
G_interaction = nets["G_interaction"]
G_follow = nets["G_follow"]
G_active = nets["G_active"]
G_follow_active = nets["G_follow_active"]
isolated = nets["isolated"]
communities = nets["communities"]
follow_communities = nets["follow_communities"]


# ============================================================
# Header + KPIs
# ============================================================
st.markdown("# YSocial Analytics")
st.caption("Interaction, network and content structure of the loaded simulation run")

posts_per_round = (
    posts.groupby(ROUND_COL).size().sort_index()
    if ROUND_COL in posts.columns else pd.Series(dtype=int)
)
reactions_per_round = (
    reactions.groupby(ROUND_COL).size().sort_index()
    if ROUND_COL in reactions.columns else pd.Series(dtype=int)
)

active_n = G_active.number_of_nodes()
total_n = G_interaction.number_of_nodes()
follow_density = nx.density(G_follow) if G_follow.number_of_nodes() > 1 else 0

# Four KPI cards with the same structure: label, value, then one quiet
# context element in the accent color (a sparkline or a thin progress bar
# with its own caption). Streamlit's built-in metric sparkline is not used:
# it takes the delta's green/red, which would make "fewer reactions than in
# the first half" look like an alarm.
KPI_CARD = _supported(st.container, border=True, height=184)


def kpi_trend_card(column, label, total, per_round, help_text):
    delta = half_split_pct_change(per_round)
    trend = [int(v) for v in per_round.tail(24)] if len(per_round) > 1 else None
    with column.container(**KPI_CARD):
        st.metric(
            label, f"{total:,}",
            delta=f"{delta:+.0f}%" if delta is not None else None,
            help=help_text,
            **_supported(st.metric, delta_description="vs. first half of the run"),
        )
        if trend:
            st.plotly_chart(interactive_sparkline(trend), **PLOTLY_WIDTH,
                            config={"displayModeBar": False, "displaylogo": False})


def kpi_share_card(column, label, value, share, bar_text, help_text):
    with column.container(**KPI_CARD):
        st.metric(label, value, help=help_text)
        st.progress(min(max(share, 0.0), 1.0), **_supported(st.progress, text=bar_text))


k1, k2, k3, k4 = st.columns(4)
kpi_trend_card(k1, "Total posts", len(posts), posts_per_round,
               "Posts and replies. The trend shows posts per round over the last 24 rounds.")
kpi_trend_card(k2, "Total reactions", len(reactions), reactions_per_round,
               "Likes and dislikes. The trend shows reactions per round over the last 24 rounds.")

n_follow = G_follow.number_of_edges()
n_seeded = sum(1 for _, _, d in G_follow.edges(data=True) if d.get("seeded"))
kpi_share_card(
    k3, "Follow ties", f"{n_follow:,}",
    n_seeded / n_follow if n_follow else 0.0,
    # One decimal, so 1,199 of 1,203 reads 99.7% rather than a rounded 100%.
    f"{n_seeded / n_follow:.1%} from the starting network" if n_follow else "No follow ties",
    "Rebuilt from the follow event log: the latest action per pair, so unfollowed ties "
    f"are not counted. Density {follow_density:.3f}. The bar shows the share of ties "
    "that come from the seeded starting network rather than from agent behaviour.",
)
pct_active = active_n / total_n if total_n else 0
kpi_share_card(
    k4, "Active agents", f"{active_n} / {total_n}", pct_active,
    f"{pct_active:.0%} interacted at least once",
    "Agents with at least one like, dislike or reply, given or received. "
    "YSocial's Admin account is not counted as an agent.",
)

# ============================================================
# Dashboard tabs
# ============================================================
# Streamlit renders :material/...: as Material Symbols (one outline icon
# set, same size and stroke), so no custom HTML or icon files are needed.
tab_names = [
    ":material/show_chart: Interactions",
    ":material/hub: Network",
    ":material/workspaces: Groups",
    ":material/sell: Topics",
    ":material/article: Text",
    ":material/forum: Comments",
    ":material/insights: Network × Content",
    ":material/compare_arrows: Compare",
]
tabs = st.tabs(tab_names)

# Plotly's default toolbar has ~12 buttons (lasso, box select, spike lines,
# compare-on-hover…) that these charts don't use. Keep only zoom, pan,
# reset and download, show it on hover, and turn off Plotly's pop-up tips.
_TOOLBAR_REMOVE = ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                   "hoverClosestCartesian", "hoverCompareCartesian"]
plotly_config = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": True,
    "responsive": True,
    "showTips": False,
    "modeBarButtonsToRemove": _TOOLBAR_REMOVE,
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "ysocial_chart"},
}



# ------------------------------------------------------------
# 1. Interactions — interactive
# ------------------------------------------------------------
with tabs[0]:
    with st.container(border=True):
        st.markdown("##### Posts and reactions per round", **_supported(
            st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))

        if len(posts_per_round) > 0:
            rounds = list(posts_per_round.index)
            reaction_values = [
                reactions_per_round.get(round_id, 0)
                for round_id in rounds
            ]

            fig = interactive_line_fig(
                rounds,
                list(posts_per_round.values),
                reaction_values,
            )
            st.plotly_chart(
                fig,
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
        else:
            st.caption("No round data available.")


# ------------------------------------------------------------
# 2. Network visualization — static, layout-focused
# ------------------------------------------------------------
with tabs[1]:
    st.markdown("### Network visualization", **_supported(
        st.markdown, help="Static on purpose: this section is about the layout. Each "
        "community gets its own region (Louvain communities, placed so that strongly tied "
        "communities sit next to each other) and its own color; inside a region agents are "
        "spread with Kamada-Kawai. Ties inside a community are drawn in its color, ties "
        "between communities in gray. Node size = degree centrality; the five most central "
        "agents are labelled. All other charts in the dashboard are interactive."))

    def _community_metrics(graph, comms, extra_label, extra_value, extra_help, weight=None):
        """Key numbers stacked in a narrow column beside the graph."""
        share, modularity = community_share_stats(graph, comms, weight=weight)
        st.metric("Agents", f"{graph.number_of_nodes():,}")
        st.metric("Edges", f"{graph.number_of_edges():,}")
        st.metric(
            "Modularity",
            f"{modularity:.3f}" if modularity is not None else "n/a",
            help="Louvain modularity: how much more ties stay inside communities than "
                 "expected by chance. Above ~0.3 is usually read as clear community structure.",
        )
        st.metric(
            "Ties inside communities",
            f"{share * 100:.0f}%" if share is not None else "n/a",
            help="Share of (undirected) ties whose two agents belong to the same community.",
        )
        st.caption(f"{extra_label}: {extra_value}  \n{extra_help}")

    def _network_card(title, caption, which, graph, comms, metrics_args):
        with st.container(border=True):
            st.markdown(f"##### {title}")
            st.caption(caption)
            if graph.number_of_edges() == 0:
                st.caption("Not enough data to draw this network.")
                return False
            plot_col, stats_col = st.columns([4, 1], **_supported(st.columns, gap="medium"))
            with plot_col:
                st.image(network_png(db_path, which), **IMAGE_WIDTH)
            with stats_col:
                _community_metrics(graph, comms, *metrics_args)
            return True

    _network_card(
        "Interaction network",
        "Edges = reactions and replies between agents (inactive agents excluded).",
        "interaction", G_active, communities,
        ("Weak components", nx.number_weakly_connected_components(G_active),
         f"density {nx.density(G_active):.4f}", "weight"),
    )

    reciprocity = nx.reciprocity(G_follow) if G_follow.number_of_edges() else None
    has_follow = _network_card(
        "Follow network",
        "Formal follow relationships, drawn without arrowheads (the graph itself is directed). "
        "In a dense follow network the ties between communities are bundled: one gray line per "
        "pair of communities, width = number of ties, strongest pairs only.",
        "follow", G_follow_active, follow_communities,
        ("Reciprocity", f"{reciprocity:.3f}" if reciprocity is not None else "n/a",
         f"density {nx.density(G_follow):.4f}"),
    )

    if has_follow:
        with st.container(border=True):
            st.markdown("##### Follow communities · summary", **_supported(
                st.markdown, help="One circle per community (area ~ number of agents). Only the "
                "strongest links between communities are drawn; line width = number of follow "
                "ties. The table lists the exact counts."))
            sum_col, table_col = st.columns([3, 2], **_supported(st.columns, gap="medium"))
            with sum_col:
                st.image(network_png(db_path, "follow_summary"), **IMAGE_WIDTH)
            with table_col:
                links = inter_community_links(G_follow_active, follow_communities)
                links_df = pd.DataFrame(
                    [{"Community A": f"C{a}", "Community B": f"C{b}", "Follow ties": n}
                     for (a, b), n in links[:15]]
                )
                st.dataframe(links_df, hide_index=True, height=400, **TABLE_WIDTH)


# ------------------------------------------------------------
# 3. Groups — interactive analytical chart
# ------------------------------------------------------------
with tabs[2]:
    with st.container(border=True):
        st.markdown("##### Communities among active agents", **_supported(
            st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))
        st.caption(
            f"Louvain community detection · {len(isolated)} inactive agents excluded · "
            f"{len(communities)} communities found"
        )

        if communities:
            sizes = sorted([len(c) for c in communities], reverse=True)
            labels = [f"C{i}" for i in range(len(sizes))]
            st.plotly_chart(
                interactive_bar_fig(labels, sizes, height=390,
                                    color=[community_color(i) for i in range(len(sizes))]),
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
        else:
            st.caption("Not enough interaction data to detect communities.")


# ------------------------------------------------------------
# 4. Topics — interactive
# ------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        st.markdown("##### Topic distribution", **_supported(st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))
        st.caption("Posts per topic id")

        pt = data["post_topics"]
        if not pt.empty and "topic_id" in pt.columns:
            topic_counts = (
                pt["topic_id"]
                .value_counts()
                .sort_values(ascending=False)
                .head(12)
            )
            labels = [f"Topic {int(topic)}" for topic in topic_counts.index]
            st.plotly_chart(
                interactive_bar_fig(
                    labels,
                    list(topic_counts.values),
                    horizontal=True,
                    height=440,
                ),
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
        else:
            st.caption("post_topics table is empty for this run.")


# ------------------------------------------------------------
# 5. Textual content — interactive
# ------------------------------------------------------------
with tabs[4]:
    c1, c2 = st.columns(2)

    with c1.container(border=True):
        st.markdown("##### Sentiment distribution", **_supported(st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))
        st.caption("VADER compound score")

        sent = data["post_sentiment"]
        if (
            not sent.empty
            and SENTIMENT_SCORE_COL in sent.columns
            and sent[SENTIMENT_SCORE_COL].notna().any()
        ):
            sentiment_values = sent[SENTIMENT_SCORE_COL].dropna()
            st.plotly_chart(
                interactive_histogram_fig(
                    sentiment_values,
                    start=-1,
                    end=1,
                    size=0.2,
                    x_title="Sentiment",
                    color=ACCENT,
                ),
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
            st.metric("Mean sentiment", f"{sentiment_values.mean():.3f}")
        else:
            st.caption("No sentiment data.")

    with c2.container(border=True):
        st.markdown("##### Toxicity distribution", **_supported(st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))
        st.caption("Perspective API score")

        tox = data["post_toxicity"]
        if (
            not tox.empty
            and TOXICITY_SCORE_COL in tox.columns
            and tox[TOXICITY_SCORE_COL].notna().any()
        ):
            toxicity_values = tox[TOXICITY_SCORE_COL].dropna()
            st.plotly_chart(
                interactive_histogram_fig(
                    toxicity_values,
                    start=0,
                    end=1,
                    size=0.1,
                    x_title="Toxicity",
                    color=DANGER,
                ),
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
            st.metric("Mean toxicity", f"{toxicity_values.mean():.3f}")
        else:
            st.caption("Toxicity annotation was off or unavailable for this run.")


# ------------------------------------------------------------
# 6. Comments — interactive
# ------------------------------------------------------------
with tabs[5]:
    with st.container(border=True):
        st.markdown("##### Most-replied posts", **_supported(st.markdown, help="Hover for exact values, drag to zoom, double-click to reset."))
        st.caption("Reply count derived from post.comment_to")

        replies = (
            posts[posts[COMMENT_TO_COL] != -1]
            if COMMENT_TO_COL in posts.columns
            else pd.DataFrame()
        )

        if not replies.empty:
            reply_counts = (
                replies.groupby(COMMENT_TO_COL)
                .size()
                .sort_values(ascending=False)
                .head(10)
            )
            labels = [f"Post {int(post_id)}" for post_id in reply_counts.index]
            st.plotly_chart(
                interactive_bar_fig(
                    labels,
                    list(reply_counts.values),
                    horizontal=True,
                    height=420,
                ),
                **PLOTLY_WIDTH,
                config=plotly_config,
            )
        else:
            st.caption("No reply relationships recorded yet.")


# ------------------------------------------------------------
# 7. Network × Content — statistics + interactive research views
# ------------------------------------------------------------
with tabs[6]:
    with st.container(border=True):
        st.markdown("##### Network × Content")
        st.caption(
            "Secondary analysis (not the thesis research questions): S1 centrality ↔ sentiment · "
            "S2 bridge vs. core tone · S3 sentiment across communities · plus follow vs. "
            "interaction centrality, the node-position part of RQ2"
        )

        if G_active.number_of_edges() > 0 and communities:
            cent = compute_centrality(db_path)
            degree_cent = cent["degree"]
            between_cent = cent["betweenness"]
            agent_to_comm = _community_map(communities)

            combined = pd.DataFrame({
                "agent_id": list(degree_cent.keys()),
                "degree_centrality": [
                    round(degree_cent[node], 4)
                    for node in degree_cent
                ],
                "betweenness_centrality": [
                    round(between_cent[node], 4)
                    for node in degree_cent
                ],
            }).sort_values("degree_centrality", ascending=False)

            combined["community_id"] = combined["agent_id"].map(agent_to_comm)

            sent = data["post_sentiment"]
            if not sent.empty and SENTIMENT_SCORE_COL in sent.columns:
                sentiment_slim = sent[["post_id", SENTIMENT_SCORE_COL]]
                posts_with_sentiment = posts.merge(
                    sentiment_slim,
                    left_on=POST_ID_COL,
                    right_on="post_id",
                    how="left",
                )

                avg_sentiment = (
                    posts_with_sentiment
                    .groupby(POST_AUTHOR_COL)[SENTIMENT_SCORE_COL]
                    .mean()
                    .reset_index()
                    .rename(columns={
                        POST_AUTHOR_COL: "agent_id",
                        SENTIMENT_SCORE_COL: "avg_sentiment",
                    })
                )
                avg_sentiment["avg_sentiment"] = (
                    avg_sentiment["avg_sentiment"].round(3)
                )
                combined = combined.merge(
                    avg_sentiment,
                    on="agent_id",
                    how="left",
                )

            median_b = combined["betweenness_centrality"].median()
            combined["role"] = np.where(
                combined["betweenness_centrality"] > median_b,
                "bridge",
                "core",
            )

            st.dataframe(
                combined.assign(
                    community_id=combined["community_id"].map(
                        lambda c: f"C{int(c)}" if pd.notna(c) else None)
                ).rename(columns={"community_id": "community"}),
                **TABLE_WIDTH,
                height=360,
                hide_index=True,
            )

            st.write("")
            r1, r2, r3, r4 = st.columns(4)

            valid = (
                combined.dropna(subset=["avg_sentiment"])
                if "avg_sentiment" in combined.columns
                else pd.DataFrame()
            )

            if len(valid) >= 3:
                rho, p = stats.spearmanr(
                    valid["degree_centrality"],
                    valid["avg_sentiment"],
                )
                r1.metric("S1 · Spearman ρ", f"{rho:.3f}", p_label(p), **NEUTRAL_DELTA)
            else:
                r1.metric("S1", "n/a", help="Needs at least 3 agents with sentiment data")

            if "avg_sentiment" in combined.columns:
                bridge = (
                    combined[combined["role"] == "bridge"]["avg_sentiment"]
                    .dropna()
                )
                core = (
                    combined[combined["role"] == "core"]["avg_sentiment"]
                    .dropna()
                )
            else:
                bridge = pd.Series(dtype=float)
                core = pd.Series(dtype=float)

            if len(bridge) >= 2 and len(core) >= 2:
                _, p2 = stats.mannwhitneyu(bridge, core)
                r2.metric(
                    "S2 · bridge − core mean",
                    f"{bridge.mean() - core.mean():+.3f}",
                    p_label(p2),
                    **NEUTRAL_DELTA,
                    help=f"Mann-Whitney U · bridge mean {bridge.mean():.3f} (n={len(bridge)}), "
                         f"core mean {core.mean():.3f} (n={len(core)})",
                )
            else:
                r2.metric("S2", "n/a")

            if "avg_sentiment" in combined.columns:
                groups = [
                    group["avg_sentiment"].dropna().values
                    for _, group in combined.groupby("community_id")
                    if len(group["avg_sentiment"].dropna()) >= 2
                ]
            else:
                groups = []

            if len(groups) >= 2:
                h3, p3 = stats.kruskal(*groups)
                r3.metric(
                    "S3 · Kruskal-Wallis H",
                    f"{h3:.2f}",
                    p_label(p3),
                    **NEUTRAL_DELTA,
                    help=f"Compared {len(groups)} communities with at least 2 agents that posted",
                )
            else:
                r3.metric("S3", "n/a")

            follow_deg = cent["follow_degree"]

            shared_agents = [
                agent
                for agent in combined["agent_id"]
                if agent in follow_deg
            ]

            if len(shared_agents) >= 3:
                interaction_series = (
                    combined
                    .set_index("agent_id")
                    .loc[shared_agents, "degree_centrality"]
                )
                follow_values = [follow_deg[agent] for agent in shared_agents]
                rho4, p4 = stats.spearmanr(
                    interaction_series,
                    follow_values,
                )
                r4.metric(
                    "Follow ↔ interaction ρ",
                    f"{rho4:.3f}",
                    p_label(p4),
                    **NEUTRAL_DELTA,
                    help=f"Spearman over {len(shared_agents)} agents present in both networks",
                )
            else:
                r4.metric("Follow ↔ interaction", "n/a")

            st.divider()
            st.markdown("##### Interactive research-question views")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                if len(valid) >= 3:
                    # Community is categorical, so one trace per community
                    # (fixed community color + legend), not a gradient scale.
                    scatter = go.Figure()
                    plot_df = valid.assign(
                        group=[f"C{int(c)}" for c in valid["community_id"]],
                        color_id=[int(c) for c in valid["community_id"]],
                    )
                    for (label, color_id), grp in plot_df.groupby(["group", "color_id"], sort=False):
                        scatter.add_trace(go.Scatter(
                            x=grp["degree_centrality"],
                            y=grp["avg_sentiment"],
                            mode="markers",
                            name=label,
                            marker=dict(size=10, color=community_color(color_id),
                                        line=dict(color=CARD_BG, width=1.5)),
                            customdata=np.stack([
                                grp["agent_id"],
                                [f"C{int(c)}" for c in grp["community_id"]],
                                grp["role"],
                            ], axis=-1),
                            hovertemplate=(
                                "Agent %{customdata[0]}"
                                "<br>Community %{customdata[1]}"
                                "<br>Role: %{customdata[2]}"
                                "<br>Degree centrality: %{x:.4f}"
                                "<br>Mean sentiment: %{y:.3f}"
                                "<extra></extra>"
                            ),
                        ))
                    # Legend in C0, C1, ... order
                    scatter.data = tuple(sorted(
                        scatter.data,
                        key=lambda t: int(t.name[1:]),
                    ))
                    scatter = style_plotly(
                        scatter,
                        height=390,
                        x_title="Interaction degree centrality",
                        y_title="Mean sentiment",
                    )
                    scatter.update_layout(legend=dict(
                        orientation="v", x=1.02, xanchor="left", y=1, yanchor="top",
                        title=dict(text="Community"),
                    ))
                    st.markdown("**S1 · Centrality and sentiment**")
                    st.plotly_chart(
                        scatter,
                        **PLOTLY_WIDTH,
                        config=plotly_config,
                    )

            with chart_col2:
                if len(bridge) > 0 and len(core) > 0:
                    role_box = go.Figure()
                    role_box.add_trace(go.Box(
                        y=core,
                        name="Core",
                        boxpoints="all",
                        jitter=0.25,
                        marker=dict(color=NEUTRAL),
                    ))
                    role_box.add_trace(go.Box(
                        y=bridge,
                        name="Bridge",
                        boxpoints="all",
                        jitter=0.25,
                        # Neutral light gray, not a palette color: bridge/core
                        # is not a community and must not look like one.
                        marker=dict(color="#D1D5DB"),
                    ))
                    role_box = style_plotly(
                        role_box,
                        height=390,
                        x_title="Agent role",
                        y_title="Mean sentiment",
                    )
                    st.markdown("**S2 · Bridge vs. core tone**")
                    st.plotly_chart(
                        role_box,
                        **PLOTLY_WIDTH,
                        config=plotly_config,
                    )

            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                if len(valid) >= 3:
                    community_box = go.Figure()
                    for community_id, group in valid.groupby("community_id"):
                        color = community_color(int(community_id))
                        community_box.add_trace(go.Box(
                            y=group["avg_sentiment"],
                            name=f"C{int(community_id)}",
                            boxpoints="all",
                            jitter=0.2,
                            marker=dict(color=color),
                            line=dict(color=color),
                            showlegend=False,
                        ))
                    community_box = style_plotly(
                        community_box,
                        height=390,
                        x_title="Community",
                        y_title="Mean sentiment",
                    )
                    st.markdown("**S3 · Sentiment by community**")
                    st.plotly_chart(
                        community_box,
                        **PLOTLY_WIDTH,
                        config=plotly_config,
                    )

            with chart_col4:
                if len(shared_agents) >= 3:
                    interaction_values = (
                        combined
                        .set_index("agent_id")
                        .loc[shared_agents, "degree_centrality"]
                        .values
                    )
                    follow_values = np.array(
                        [follow_deg[agent] for agent in shared_agents]
                    )

                    r4_scatter = go.Figure(go.Scatter(
                        x=follow_values,
                        y=interaction_values,
                        mode="markers",
                        marker=dict(color=ACCENT, size=9),
                        text=[str(agent) for agent in shared_agents],
                        hovertemplate=(
                            "Agent %{text}"
                            "<br>Follow centrality: %{x:.4f}"
                            "<br>Interaction centrality: %{y:.4f}"
                            "<extra></extra>"
                        ),
                    ))
                    r4_scatter = style_plotly(
                        r4_scatter,
                        height=390,
                        x_title="Follow degree centrality",
                        y_title="Interaction degree centrality",
                    )
                    st.markdown("**Follow vs. interaction centrality**")
                    st.plotly_chart(
                        r4_scatter,
                        **PLOTLY_WIDTH,
                        config=plotly_config,
                    )

        else:
            st.caption("Not enough interaction data for the combined analysis.")


# ------------------------------------------------------------
# 8. Compare
# ------------------------------------------------------------
with tabs[7]:
    with st.container(border=True):
        st.markdown("##### Compare runs")
        st.caption("Runs stored side by side (one folder per run) compared on the thesis "
                   "network measures")

        if single_file_mode:
            st.caption(
                "An uploaded database is loaded, so there are no other runs next to it to "
                "compare. Type a local path in the sidebar instead."
            )
        else:
            parent, runs = find_runs(db_path)
            if len(runs) < 2:
                st.caption(
                    f"Only this run was found in `{parent}`. To compare runs, put each one in "
                    "its own folder next to this run's folder, with its database_server.db "
                    "inside (for example `data/Pilot_k20/database_server.db`). YSocial's own "
                    "experiments folder already has this layout."
                )
            else:
                rows, failed = [], []
                with st.spinner(f"Computing network measures for {len(runs)} runs…"):
                    for r in runs:
                        try:
                            rows.append({"run": r["name"], "loaded": r["current"],
                                         **run_metrics(r["path"], r["mtime"])})
                        except Exception as e:  # a broken or unfinished database
                            failed.append(f"{r['name']} ({e.__class__.__name__})")
                table = pd.DataFrame(rows)
                PCT = {"active share", "volume on follow ties", "followed share of timeline"}

                def fmt_value(metric, v):
                    if pd.isna(v):
                        return "–"
                    if metric in PCT:
                        return f"{v:.1%}"
                    if metric == "modularity":
                        return f"{v:.3f}"
                    if isinstance(v, float) and not float(v).is_integer():
                        return f"{v:,.2f}"
                    return f"{int(v):,}"

                # One column per run, one row per measure: easier to read than a
                # wide table when there are only a handful of runs.
                metric_names = [c for c in table.columns if c not in ("run", "loaded")]
                labels = [f"{r} (loaded)" if cur else r for r, cur in zip(table["run"], table["loaded"])]
                wide = pd.DataFrame(
                    {lab: [fmt_value(m, table.loc[i, m]) for m in metric_names]
                     for i, lab in enumerate(labels)},
                    index=pd.Index(metric_names, name="measure"),
                )
                st.dataframe(wide, **TABLE_WIDTH, height=35 * (len(wide) + 1) + 2)
                st.caption(
                    f"{len(rows)} runs in `{parent}`. Shares and rates can be compared across runs "
                    "of different length; counts can't. Network measures use the same rules as the "
                    "thesis scripts."
                    + (f" Skipped: {', '.join(failed)}." if failed else "")
                )

                if len(rows) > 1:
                    metric = st.selectbox("Compare runs on", metric_names,
                                          index=metric_names.index("followed share of timeline"))
                    raw = table[metric]
                    values = raw * 100 if metric in PCT else raw
                    bar = go.Figure(go.Bar(
                        x=table["run"], y=values.fillna(0),
                        marker=dict(color=[ACCENT if cur else MUTED for cur in table["loaded"]]),
                        text=[fmt_value(metric, v) if pd.notna(v) else "no data" for v in raw],
                        textposition="outside", textfont=dict(color=TEXT2), cliponaxis=False,
                        hovertemplate="%{x}<br>" + metric + ": %{text}<extra></extra>",
                    ))
                    bar = style_plotly(bar, height=340,
                                       y_title=metric + (" (%)" if metric in PCT else ""))
                    bar.update_layout(showlegend=False, bargap=0.45)
                    st.plotly_chart(bar, **PLOTLY_WIDTH, config=plotly_config)
                    st.caption("Purple = the run loaded in the dashboard. \"no data\" = the run "
                               "has no recommendations table (older runs).")

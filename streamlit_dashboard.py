"""
YSocial Analytics Dashboard
==================================================================

LOCAL RUN:
    pip install -r requirements.txt
    streamlit run streamlit_dashboard.py


"""

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
import streamlit as st

# ============================================================
# Verified schema
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

# Premium dark-mode palette — matches .streamlit/config.toml
ACCENT = "#8B7FE8"
NEUTRAL = "#6B7280"
SUCCESS = "#34D399"   # pastel emerald, not harsh green
DANGER = "#F87171"    # pastel red, not harsh red
TEXT2 = "#9CA3AF"
CARD_BG = "#1A1D27"

st.set_page_config(page_title="YSocial Analytics", layout="wide", initial_sidebar_state="expanded")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.edgecolor": "#2D3140",
    "axes.labelcolor": TEXT2,
    "text.color": "#E5E7EB",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "figure.facecolor": CARD_BG,
    "axes.facecolor": CARD_BG,
    "savefig.facecolor": CARD_BG,
})


# ============================================================
# Chart helpers — matplotlib 
# ============================================================
def _clean_axes(ax, horizontal=False):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#2D3140")
    ax.grid(axis="x" if horizontal else "y", color="#242836", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _smooth(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return np.arange(len(y)), y
    try:
        from scipy.interpolate import make_interp_spline
        x = np.arange(len(y))
        xnew = np.linspace(0, len(y) - 1, max(len(y) * 10, 60))
        spline = make_interp_spline(x, y, k=min(3, len(y) - 1))
        return xnew, spline(xnew)
    except Exception:
        return np.arange(len(y)), y


def gradient_area_line(ax, y, color=ACCENT, linewidth=2.6, layers=24, label=None):
    """A smooth line with a soft, layered gradient fill underneath —
    the plain-matplotlib equivalent of a CSS linear-gradient area chart."""
    xnew, ynew = _smooth(y)
    baseline = min(0, float(np.min(ynew)))
    ax.plot(xnew, ynew, color=color, linewidth=linewidth, solid_capstyle="round",
             solid_joinstyle="round", label=label, zorder=3)
    for i in range(layers):
        frac_top = 1 - i / layers
        frac_bot = 1 - (i + 1) / layers
        y_top = baseline + (ynew - baseline) * frac_top
        y_bot = baseline + (ynew - baseline) * frac_bot
        alpha = 0.20 * (1 - i / layers)
        ax.fill_between(xnew, y_bot, y_top, color=color, linewidth=0, alpha=alpha, zorder=2)


def line_fig(labels, y1, y2=None, label1="Posts", label2="Reactions", figsize=(9, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    gradient_area_line(ax, y1, color=ACCENT, label=label1)
    if y2 is not None:
        xnew2, ynew2 = _smooth(y2)
        ax.plot(xnew2, ynew2, color=NEUTRAL, linewidth=1.8, linestyle=(0, (4, 3)),
                 solid_capstyle="round", label=label2, zorder=3)
        ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=TEXT2)
    n = len(labels)
    step = max(1, n // 10)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([labels[i] for i in range(0, n, step)])
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def rounded_bar_fig(labels, values, horizontal=False, figsize=(9, 3.4), color=ACCENT):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(values)
    if n == 0:
        ax.axis("off")
        return fig
    vmax = max(values) if max(values) > 0 else 1

    if horizontal:
        thickness = 0.6
        for i, (lab, v) in enumerate(zip(labels, values)):
            w = max(v, vmax * 0.012)
            r = thickness * 0.5
            box = FancyBboxPatch((0, i - thickness / 2), w, thickness,
                                  boxstyle=f"round,pad=0,rounding_size={min(r, w*0.4):.3f}",
                                  linewidth=0, facecolor=color, alpha=0.92)
            ax.add_patch(box)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(0, vmax * 1.15)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        width = 0.55
        for i, (lab, v) in enumerate(zip(labels, values)):
            h = max(v, vmax * 0.012)
            r = width * 0.5
            box = FancyBboxPatch((i - width / 2, 0), width, h,
                                  boxstyle=f"round,pad=0,rounding_size={min(r, h*0.25):.3f}",
                                  linewidth=0, facecolor=color, alpha=0.92)
            ax.add_patch(box)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(0, vmax * 1.15)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45 if n > 8 else 0, ha="right" if n > 8 else "center", fontsize=8.5)

    _clean_axes(ax, horizontal=horizontal)
    fig.tight_layout()
    return fig


def sparkline_fig(values, color=ACCENT, figsize=(1.7, 0.46)):
    fig, ax = plt.subplots(figsize=figsize)
    if values and len(values) > 1:
        gradient_area_line(ax, values, color=color, linewidth=1.8, layers=14)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def network_fig(G, centrality, figsize=(8.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    if G.number_of_nodes() == 0:
        ax.axis("off")
        return fig
    pos = nx.spring_layout(G, seed=42, k=0.65)
    vals = list(centrality.values()) or [0]
    maxc = max(vals) or 1
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    sizes = np.array([centrality.get(n, 0) / maxc for n in G.nodes()])
    node_sizes = 70 + 650 * sizes

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#2D3140", width=0.8)
    # soft glow: a larger, faint halo behind each node, then the solid node
    ax.scatter(xs, ys, s=node_sizes * 3.4, c=ACCENT, alpha=0.10, linewidths=0, zorder=2)
    ax.scatter(xs, ys, s=node_sizes, c=[ACCENT if s > 0.45 else NEUTRAL for s in sizes],
               alpha=0.92, linewidths=0, zorder=3)
    ax.axis("off")
    fig.tight_layout()
    return fig


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
# Data loading — 
# ============================================================
@st.cache_data(ttl=30)
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    data = {
        "posts": pd.read_sql("SELECT * FROM post;", conn),
        "reactions": pd.read_sql("SELECT * FROM reactions;", conn),
        "follow": pd.read_sql("SELECT * FROM follow;", conn),
        "users": pd.read_sql("SELECT * FROM user_mgmt;", conn),
    }
    for table in ["post_sentiment", "post_toxicity", "post_topics"]:
        try:
            data[table] = pd.read_sql(f"SELECT * FROM {table};", conn)
        except Exception:
            data[table] = pd.DataFrame()
    conn.close()
    return data


@st.cache_data(ttl=30)
def scan_all_experiments(current_db_path):
    experiments_dir = os.path.dirname(os.path.dirname(current_db_path))
    rows = []
    for exp_folder in sorted(glob.glob(os.path.join(experiments_dir, "*"))):
        db_file = os.path.join(exp_folder, "database_server.db")
        if not os.path.isfile(db_file):
            continue
        name = os.path.basename(exp_folder)
        config_path = os.path.join(exp_folder, "config_server.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    name = json.load(f).get("name", name)
            except Exception:
                pass
        try:
            c = sqlite3.connect(db_file)
            counts = {}
            for table in ["post", "reactions", "follow", "user_mgmt"]:
                try:
                    counts[table] = pd.read_sql(f"SELECT COUNT(*) n FROM {table};", c)["n"].iloc[0]
                except Exception:
                    counts[table] = None
            c.close()
            rows.append({"name": name, "folder": os.path.basename(exp_folder),
                         "posts": counts["post"], "reactions": counts["reactions"],
                         "follow": counts["follow"], "agents": counts["user_mgmt"],
                         "current": db_file == current_db_path})
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_interaction_network(posts, reactions, users):
    post_author = dict(zip(posts[POST_ID_COL], posts[POST_AUTHOR_COL]))
    G = nx.DiGraph()
    id_col = "id" if "id" in users.columns else users.columns[0]
    G.add_nodes_from(users[id_col])
    for _, row in reactions.iterrows():
        actor, target = row.get(REACTION_ACTOR_COL), row.get(REACTION_TARGET_COL)
        author = post_author.get(target)
        if author is not None and author != actor:
            w = G[actor][author]["weight"] + 1 if G.has_edge(actor, author) else 1
            G.add_edge(actor, author, weight=w)
    if COMMENT_TO_COL in posts.columns:
        for _, row in posts.iterrows():
            parent = row.get(COMMENT_TO_COL)
            if parent is not None and parent != -1:
                author = post_author.get(parent)
                actor = row.get(POST_AUTHOR_COL)
                if author is not None and author != actor:
                    w = G[actor][author]["weight"] + 1 if G.has_edge(actor, author) else 1
                    G.add_edge(actor, author, weight=w)
    return G


def build_follow_network(follow, users):
    G = nx.DiGraph()
    id_col = "id" if "id" in users.columns else users.columns[0]
    G.add_nodes_from(users[id_col])
    for _, row in follow.iterrows():
        s, t = row.get(FOLLOW_SOURCE_COL), row.get(FOLLOW_TARGET_COL)
        if s is not None and t is not None:
            G.add_edge(s, t)
    return G


# ============================================================
# Sidebar and cloud-friendly data selection
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
    help="Optional: upload a different database_server.db file for this browser session.",
)

if uploaded_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    db_path = tmp.name
    single_file_mode = True
    st.sidebar.success("Uploaded database loaded")
elif DEFAULT_DB.is_file():
    db_path = str(DEFAULT_DB)
    single_file_mode = False
    st.sidebar.success("Thesis_scaled_v1 loaded")
else:
    st.sidebar.info(
        "The online demo database has not been added yet. "
        "Upload a database_server.db file above to explore the dashboard."
    )
    st.markdown("## YSocial Analytics")
    st.warning("No simulation database is available yet.")
    st.write(
        "Use the sidebar to upload a YSocial `database_server.db` file. "
        "The dashboard reads the file for analysis and does not modify it."
    )
    st.stop()

try:
    data = load_data(db_path)
except Exception as e:
    st.error(f"Could not open this database: {e}")
    st.stop()

posts, reactions, follow, users = data["posts"], data["reactions"], data["follow"], data["users"]
G_interaction = build_interaction_network(posts, reactions, users)
G_follow = build_follow_network(follow, users)
isolated = [n for n in G_interaction.nodes() if G_interaction.degree(n) == 0]
G_active = G_interaction.copy()
G_active.remove_nodes_from(isolated)
G_active_und = G_active.to_undirected()
communities = list(nx.algorithms.community.louvain_communities(G_active_und, seed=42)) if G_active.number_of_edges() > 0 else []

# ============================================================
# Header + KPI row
# ============================================================
st.markdown("## YSocial Analytics")
st.caption("Interaction, follow, and content structure for the loaded simulation run")
st.write("")

posts_per_round = posts.groupby(ROUND_COL).size().sort_index() if ROUND_COL in posts.columns else pd.Series(dtype=int)
reactions_per_round = reactions.groupby(ROUND_COL).size().sort_index() if ROUND_COL in reactions.columns else pd.Series(dtype=int)
active_n = G_active.number_of_nodes()
total_n = G_interaction.number_of_nodes()
density = nx.density(G_follow) if G_follow.number_of_nodes() > 1 else 0

k1, k2, k3, k4 = st.columns(4)

with k1.container(border=True):
    d = half_split_pct_change(posts_per_round)
    st.metric("Total posts", f"{len(posts):,}", delta=f"{d:+.0f}% vs. first half" if d is not None else None)
    if len(posts_per_round) > 1:
        st.pyplot(sparkline_fig(list(posts_per_round.tail(24))), use_container_width=False)

with k2.container(border=True):
    d = half_split_pct_change(reactions_per_round)
    st.metric("Total reactions", f"{len(reactions):,}", delta=f"{d:+.0f}% vs. first half" if d is not None else None)
    if len(reactions_per_round) > 1:
        st.pyplot(sparkline_fig(list(reactions_per_round.tail(24)), color="#B0A5F0"), use_container_width=False)

with k3.container(border=True):
    st.metric("Follow edges", f"{G_follow.number_of_edges():,}")
    st.caption(f"density {density:.3f} · static after seeding")
    st.progress(min(density * 20, 1.0))

with k4.container(border=True):
    pct_active = active_n / total_n if total_n else 0
    st.metric("Active agents", f"{active_n} / {total_n}")
    st.caption(f"{pct_active*100:.0f}% of registered agents ever acted")
    st.progress(pct_active)

st.write("")

# ============================================================
# Tabs
# ============================================================
tab_names = ["📈 Interactions", "🔗 Friendship network", "🧩 Groups", "🏷️ Topics",
             "💬 Textual content", "🗨️ Comments", "⭐ Flagship", "📊 Compare"]
tabs = st.tabs(tab_names)

with tabs[0]:
    with st.container(border=True):
        st.markdown("##### Posts and reactions per round")
        st.caption("Volume over the simulation horizon")
        st.write("")
        if len(posts_per_round) > 0:
            x = list(posts_per_round.index)
            y2 = [reactions_per_round.get(r, 0) for r in x]
            st.pyplot(line_fig(x, list(posts_per_round.values), y2))
        else:
            st.caption("No round data available.")

with tabs[1]:
    with st.container(border=True):
        st.markdown("##### Follow graph")
        st.caption(f"{G_follow.number_of_edges()} edges across {G_follow.number_of_nodes()} agents")
        st.write("")
        if G_follow.number_of_edges() > 0:
            deg = nx.degree_centrality(G_follow)
            active_follow = G_follow.subgraph([n for n in G_follow.nodes() if G_follow.degree(n) > 0])
            st.pyplot(network_fig(active_follow, deg))
            c1, c2, c3 = st.columns(3)
            c1.metric("Density", f"{nx.density(G_follow):.4f}")
            c2.metric("Connected components", nx.number_weakly_connected_components(G_follow))
            c3.metric("Reciprocity", f"{nx.reciprocity(G_follow):.3f}")
        else:
            st.caption("No follow edges in this run — expected under default simulation parameters "
                       "(see Methodology §6.1). The Groups and Flagship tabs fall back to the interaction network instead.")

with tabs[2]:
    with st.container(border=True):
        st.markdown("##### Communities")
        st.caption(f"Louvain, active agents only · excluded {len(isolated)} inactive agents out of {total_n} "
                   f"· {len(communities)} communities found")
        st.write("")
        if communities:
            sizes = sorted([len(c) for c in communities], reverse=True)
            labels = [f"C{i}" for i in range(len(sizes))]
            st.pyplot(rounded_bar_fig(labels, sizes, figsize=(9, 3.2)))
        else:
            st.caption("Not enough interaction data to detect communities yet.")

with tabs[3]:
    with st.container(border=True):
        st.markdown("##### Topic distribution")
        st.caption("Post count per topic id")
        st.write("")
        pt = data["post_topics"]
        if not pt.empty and "topic_id" in pt.columns:
            tc = pt["topic_id"].value_counts().sort_values(ascending=False).head(12)
            labels = [f"Topic {int(t)}" for t in tc.index]
            st.pyplot(rounded_bar_fig(labels, list(tc.values), horizontal=True, figsize=(9, 4.2)))
        else:
            st.caption("post_topics table is empty for this run.")

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("##### Sentiment distribution")
        st.caption("VADER compound score")
        st.write("")
        sent = data["post_sentiment"]
        if not sent.empty and SENTIMENT_SCORE_COL in sent.columns:
            bins = np.linspace(-1, 1, 11)
            hist, edges_ = np.histogram(sent[SENTIMENT_SCORE_COL].dropna(), bins=bins)
            labels = [f"{edges_[i]:.1f}" for i in range(len(edges_) - 1)]
            st.pyplot(rounded_bar_fig(labels, list(hist), figsize=(6, 3.2)))
            st.metric("Mean sentiment", f"{sent[SENTIMENT_SCORE_COL].mean():.3f}")
        else:
            st.caption("No sentiment data.")
    with c2.container(border=True):
        st.markdown("##### Toxicity distribution")
        st.caption("Perspective API score")
        st.write("")
        tox = data["post_toxicity"]
        if not tox.empty and TOXICITY_SCORE_COL in tox.columns and tox[TOXICITY_SCORE_COL].notna().any():
            bins = np.linspace(0, 1, 11)
            hist, edges_ = np.histogram(tox[TOXICITY_SCORE_COL].dropna(), bins=bins)
            labels = [f"{edges_[i]:.1f}" for i in range(len(edges_) - 1)]
            st.pyplot(rounded_bar_fig(labels, list(hist), figsize=(6, 3.2), color=DANGER))
        else:
            st.caption("Toxicity annotation was off for this run.")

with tabs[5]:
    with st.container(border=True):
        st.markdown("##### Most-replied posts")
        st.caption("Reply count via comment_to")
        st.write("")
        replies = posts[posts[COMMENT_TO_COL] != -1] if COMMENT_TO_COL in posts.columns else pd.DataFrame()
        if not replies.empty:
            tc = replies.groupby(COMMENT_TO_COL).size().sort_values(ascending=False).head(10)
            labels = [f"Post {int(p)}" for p in tc.index]
            st.pyplot(rounded_bar_fig(labels, list(tc.values), horizontal=True, figsize=(9, 3.8)))
        else:
            st.caption("No reply relationships recorded yet.")

with tabs[6]:
    with st.container(border=True):
        st.markdown("##### Network × content — flagship module")
        st.caption("RQ1 centrality↔sentiment · RQ2 bridge vs. core · RQ3 cross-community · RQ4 social vs. content centrality")
        st.write("")

        if G_active.number_of_edges() > 0 and communities:
            degree_cent = nx.degree_centrality(G_active)
            between_cent = nx.betweenness_centrality(G_active)
            agent_to_comm = {a: i for i, c in enumerate(communities) for a in c}

            combined = pd.DataFrame({
                "agent_id": list(degree_cent.keys()),
                "degree_centrality": [round(v, 4) for v in degree_cent.values()],
                "betweenness_centrality": [round(between_cent[n], 4) for n in degree_cent.keys()],
            }).sort_values("degree_centrality", ascending=False)
            combined["community_id"] = combined["agent_id"].map(agent_to_comm)

            sent = data["post_sentiment"]
            if not sent.empty and SENTIMENT_SCORE_COL in sent.columns:
                sslim = sent[["post_id", SENTIMENT_SCORE_COL]]
                pw = posts.merge(sslim, left_on=POST_ID_COL, right_on="post_id", how="left")
                avg_sent = (pw.groupby(POST_AUTHOR_COL)[SENTIMENT_SCORE_COL].mean().reset_index()
                            .rename(columns={POST_AUTHOR_COL: "agent_id", SENTIMENT_SCORE_COL: "avg_sentiment"}))
                avg_sent["avg_sentiment"] = avg_sent["avg_sentiment"].round(3)
                combined = combined.merge(avg_sent, on="agent_id", how="left")

            median_b = combined["betweenness_centrality"].median()
            combined["role"] = np.where(combined["betweenness_centrality"] > median_b, "bridge", "core")

            st.dataframe(combined, use_container_width=True, height=420, hide_index=True)
            st.caption("Click any column header to sort.")
            st.write("")

            r1, r2, r3, r4 = st.columns(4)

            valid = combined.dropna(subset=["avg_sentiment"]) if "avg_sentiment" in combined.columns else pd.DataFrame()
            if len(valid) >= 3:
                rho, p = stats.spearmanr(valid["degree_centrality"], valid["avg_sentiment"])
                r1.metric("RQ1 · Spearman ρ", f"{rho:.3f}", f"p = {p:.3f}")
            else:
                r1.metric("RQ1", "n/a")

            if "avg_sentiment" in combined.columns:
                bridge = combined[combined.role == "bridge"]["avg_sentiment"].dropna()
                core = combined[combined.role == "core"]["avg_sentiment"].dropna()
            else:
                bridge, core = pd.Series(dtype=float), pd.Series(dtype=float)
            if len(bridge) >= 2 and len(core) >= 2:
                u, p2 = stats.mannwhitneyu(bridge, core)
                r2.metric("RQ2 · Mann-Whitney p", f"{p2:.3f}", f"bridge {bridge.mean():.2f} vs core {core.mean():.2f}")
            else:
                r2.metric("RQ2", "n/a")

            if "avg_sentiment" in combined.columns:
                groups = [g["avg_sentiment"].dropna().values for _, g in combined.groupby("community_id")
                          if len(g["avg_sentiment"].dropna()) >= 2]
            else:
                groups = []
            if len(groups) >= 2:
                h, p3 = stats.kruskal(*groups)
                r3.metric("RQ3 · Kruskal-Wallis p", f"{p3:.3f}", f"{len(groups)} communities")
            else:
                r3.metric("RQ3", "n/a")

            if G_follow.number_of_edges() > 0:
                follow_deg = nx.degree_centrality(G_follow)
                shared = [a for a in combined["agent_id"] if a in follow_deg]
                if len(shared) >= 3:
                    interaction_vals = combined.set_index("agent_id").loc[shared, "degree_centrality"]
                    follow_vals = [follow_deg[a] for a in shared]
                    rho4, p4 = stats.spearmanr(interaction_vals, follow_vals)
                    r4.metric("RQ4 · social↔content ρ", f"{rho4:.3f}", f"p = {p4:.3f}")
                else:
                    r4.metric("RQ4", "n/a")
            else:
                r4.metric("RQ4", "no follow data")
        else:
            st.caption("Not enough interaction data for the flagship analysis yet.")

with tabs[7]:
    with st.container(border=True):
        st.markdown("##### Compare experiments")
        st.caption("All experiment folders under the same directory")
        st.write("")
        if single_file_mode:
            st.caption("Uploaded a single file — switch to the local path option to compare multiple experiment folders.")
        else:
            cmp_df = scan_all_experiments(db_path)
            if cmp_df.empty:
                st.caption("No sibling experiment folders found.")
            else:
                st.dataframe(cmp_df[["name", "posts", "reactions", "follow", "agents", "current"]],
                            use_container_width=True, height=300, hide_index=True)

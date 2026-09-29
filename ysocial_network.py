"""
ysocial_network.py — shared network construction for the thesis analysis
==========================================================================

One place that defines how the two layers are built from a YSocial
database, so the analysis pipeline, the dashboard and later phases all use
exactly the same rules.

Verified against the YSocial source code (YServer / YClient, Sept 2026):

Follow table (event log, not a list of ties)
    - Each row is one action: action = "follow" or "unfollow".
    - `user_id` is the agent who follows; `follower_id` is the agent being
      followed (the column name is misleading).
    - Seed-network ties are written at round 0. YSocial writes every
      undirected seed edge in BOTH directions, so the seeded follow layer
      has reciprocity 1 by construction.
    - A tie exists at a given moment if the latest action for that
      (follower, followee) pair is "follow".

Interaction layer
    - Directed edge actor -> author, for every reaction (like / dislike) to
      one of the author's posts and every reply (post.comment_to) to one.
    - Weight = number of such interactions. Self-interactions are dropped.

Time
    - One round = one simulated hour. The `rounds` table maps each round
      id to (day, hour); if it is missing, day = round // 24 is used.

Exposure
    - The `recommendations` table stores, for each timeline read, the agent
      and the pipe-separated list of post ids it was shown in that round.
    - Mention notifications are not logged there, so a few interactions may
      have no matching exposure; phase1_validate.py measures how many.
"""

import sqlite3

import networkx as nx
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Schema (verified against YServer's models)
# ------------------------------------------------------------------
POST_ID, POST_AUTHOR, POST_PARENT, POST_ROUND = "id", "user_id", "comment_to", "round"
REACT_ACTOR, REACT_POST, REACT_TYPE, REACT_ROUND = "user_id", "post_id", "type", "round"
FOLLOW_FROM, FOLLOW_TO, FOLLOW_ACTION, FOLLOW_ROUND = "user_id", "follower_id", "action", "round"
SEED_ROUND = 0


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def _table_exists(conn, name):
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None


def load_tables(db_path):
    """Read the tables the analysis needs. The database is opened
    read-only, so the analysis can never modify simulation data."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for name in ["post", "reactions", "follow", "user_mgmt"]:
            out[name] = pd.read_sql(f"SELECT * FROM {name}", conn)
        for name in ["rounds", "recommendations"]:
            out[name] = (pd.read_sql(f"SELECT * FROM {name}", conn)
                         if _table_exists(conn, name) else pd.DataFrame())
    finally:
        conn.close()
    return out


def admin_ids(users):
    """YSocial inserts an 'Admin' account (email admin@y-not.social) into
    every new experiment database. It is the platform's own account, not a
    simulated agent."""
    mask = users["username"].astype(str).eq("Admin")
    if "email" in users.columns:
        mask &= users["email"].astype(str).eq("admin@y-not.social") | users["email"].isna()
    return sorted(users.loc[mask, "id"].dropna().astype(int).tolist())


def node_universe(users, include_pages=True, include_admin=False):
    """All simulated agents. The YSocial Admin account is excluded by
    default; news pages (user_mgmt.is_page = 1) can be excluded too."""
    u = users
    if not include_admin:
        u = u[~u["id"].isin(admin_ids(users))]
    if not include_pages and "is_page" in u.columns:
        u = u[u["is_page"].fillna(0).astype(int) == 0]
    return sorted(u["id"].dropna().astype(int).tolist())


def round_calendar(rounds):
    """round id -> (day, hour). Falls back to 24 rounds per day."""
    if rounds is not None and not rounds.empty and {"id", "day", "hour"} <= set(rounds.columns):
        return rounds[["id", "day", "hour"]].rename(columns={"id": "round"})
    return None


def round_to_day(round_ids, calendar):
    r = pd.Series(round_ids).astype(int)
    if calendar is None:
        return (r // 24).to_numpy()
    lookup = dict(zip(calendar["round"], calendar["day"]))
    return r.map(lookup).fillna(r // 24).astype(int).to_numpy()


# ------------------------------------------------------------------
# Follow layer (event log)
# ------------------------------------------------------------------
def follow_events(follow):
    """Clean, ordered follow log with explicit column names."""
    ev = follow.rename(columns={FOLLOW_FROM: "follower", FOLLOW_TO: "followee",
                                FOLLOW_ACTION: "action", FOLLOW_ROUND: "round"})
    cols = ["follower", "followee", "action", "round"] + (["id"] if "id" in ev.columns else [])
    ev = ev[cols].dropna(subset=["follower", "followee", "action", "round"]).copy()
    ev[["follower", "followee", "round"]] = ev[["follower", "followee", "round"]].astype(int)
    ev["action"] = ev["action"].str.strip().str.lower()
    ev = ev[ev["follower"] != ev["followee"]]
    order = ["round", "id"] if "id" in ev.columns else ["round"]
    return ev.sort_values(order, kind="stable").reset_index(drop=True)


def follow_state(follow, before_round=None):
    """Active (follower, followee) pairs.

    before_round=None -> final state. Otherwise the state at the START of
    that round: only actions with round < before_round count. A pair is
    active if its latest action is 'follow'."""
    ev = follow_events(follow)
    if before_round is not None:
        ev = ev[ev["round"] < before_round]
    latest = ev.groupby(["follower", "followee"], sort=False).tail(1)
    return latest[latest["action"] == "follow"][["follower", "followee", "round"]] \
        .rename(columns={"round": "since_round"}).reset_index(drop=True)


def follow_layer(follow, nodes, before_round=None):
    """Directed follow graph (follower -> followee) on a fixed node set."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    state = follow_state(follow, before_round)
    state = state[state["follower"].isin(nodes) & state["followee"].isin(nodes)]
    G.add_edges_from((int(a), int(b), {"since_round": int(r), "seeded": int(r) == SEED_ROUND})
                     for a, b, r in state.itertuples(index=False, name=None))
    return G


# ------------------------------------------------------------------
# Interaction layer
# ------------------------------------------------------------------
def interaction_events(posts, reactions):
    """One row per interaction: actor, author, target post, round, kind.
    kind is 'like', 'dislike' (reactions) or 'reply' (comment_to)."""
    authors = posts[[POST_ID, POST_AUTHOR]].rename(columns={POST_ID: "post_id", POST_AUTHOR: "author"})

    r = reactions[[REACT_ACTOR, REACT_POST, REACT_ROUND] + ([REACT_TYPE] if REACT_TYPE in reactions else [])]
    r = r.rename(columns={REACT_ACTOR: "actor", REACT_POST: "post_id", REACT_ROUND: "round",
                          REACT_TYPE: "kind"})
    if "kind" not in r:
        r = r.assign(kind="reaction")
    r = r.merge(authors, on="post_id", how="left")

    parents = posts[posts[POST_PARENT].notna() & (posts[POST_PARENT] != -1)]
    c = parents[[POST_AUTHOR, POST_PARENT, POST_ROUND]].rename(
        columns={POST_AUTHOR: "actor", POST_PARENT: "post_id", POST_ROUND: "round"}).assign(kind="reply")
    c = c.merge(authors, on="post_id", how="left")

    ev = pd.concat([r, c], ignore_index=True)
    ev["kind"] = ev["kind"].astype(str).str.lower()
    ev["target_found"] = ev["author"].notna()
    return ev


def interaction_layer(events, nodes, until_round=None):
    """Directed, weighted interaction graph (actor -> author) on a fixed
    node set. until_round (inclusive) builds a cumulative snapshot."""
    ev = events[events["target_found"]].copy()
    if until_round is not None:
        ev = ev[ev["round"] <= until_round]
    ev[["actor", "author"]] = ev[["actor", "author"]].astype(int)
    ev = ev[(ev["actor"] != ev["author"]) & ev["actor"].isin(nodes) & ev["author"].isin(nodes)]
    w = ev.groupby(["actor", "author"]).size()
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from((int(a), int(b), int(n)) for (a, b), n in w.items())
    return G


def active_agents(G_interaction):
    return sorted(n for n in G_interaction.nodes() if G_interaction.degree(n) > 0)


def undirected_projection(G):
    """Undirected graph; weights of the two directions are summed."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if H.has_edge(u, v):
            H[u][v]["weight"] += w
        else:
            H.add_edge(u, v, weight=w)
    return H


# ------------------------------------------------------------------
# Exposure (recommendations table)
# ------------------------------------------------------------------
def exposures(recommendations, posts):
    """One row per (agent, post) at its FIRST exposure: agent, post_id,
    author, round. Repeated showings of the same post to the same agent
    count once."""
    if recommendations is None or recommendations.empty:
        return pd.DataFrame(columns=["agent", "post_id", "author", "round"])
    rec = recommendations[["user_id", "post_ids", "round"]].dropna().copy()
    rec["post_id"] = rec["post_ids"].astype(str).str.split("|")
    rec = rec.explode("post_id")
    rec = rec[rec["post_id"].str.strip().str.fullmatch(r"-?\d+")]
    rec = rec.assign(agent=rec["user_id"].astype(int), post_id=rec["post_id"].astype(int),
                     round=rec["round"].astype(int))[["agent", "post_id", "round"]]
    first = rec.groupby(["agent", "post_id"], as_index=False)["round"].min()
    authors = posts[[POST_ID, POST_AUTHOR]].rename(columns={POST_ID: "post_id", POST_AUTHOR: "author"})
    return first.merge(authors, on="post_id", how="left")


def attach_follow_status(exp, follow):
    """Adds `followed`: did the agent follow the post's author at the start
    of the exposure round? Uses the follow log, not the final network."""
    exp = exp.dropna(subset=["author"]).copy()
    exp["author"] = exp["author"].astype(int)
    ev = follow_events(follow).rename(columns={"follower": "agent", "followee": "author",
                                               "round": "action_round"})
    if ev.empty or exp.empty:
        return exp.assign(followed=False)
    # State at the start of round t = latest action with action_round < t.
    exp = exp.assign(_t=exp["round"] - 1).sort_values("_t", kind="stable")
    ev = ev.sort_values("action_round", kind="stable")
    merged = pd.merge_asof(exp, ev[["agent", "author", "action_round", "action"]],
                           left_on="_t", right_on="action_round",
                           by=["agent", "author"], direction="backward")
    merged["followed"] = merged["action"].eq("follow")
    return merged.drop(columns=["_t", "action_round", "action"]).reset_index(drop=True)


def timeline_reads(recommendations, posts, follow):
    """Every timeline read, post by post: one row per (read, shown post) with
    read_id, agent, post_id, round, author and `followed` (did the agent
    follow the author at the start of that round?). One recommendations row
    is one read. Unlike exposures(), repeated showings are kept, because the
    question here is what each timeline was made of. Own posts are dropped."""
    if recommendations is None or recommendations.empty:
        return pd.DataFrame(columns=["read_id", "agent", "post_id", "round", "author", "followed"])
    rec = recommendations[["user_id", "post_ids", "round"]].dropna().reset_index(drop=True)
    rec["read_id"] = np.arange(len(rec))
    rec["post_id"] = rec["post_ids"].astype(str).str.split("|")
    rec = rec.explode("post_id")
    rec = rec[rec["post_id"].str.strip().str.fullmatch(r"-?\d+")]
    rec = rec.assign(agent=rec["user_id"].astype(int), post_id=rec["post_id"].astype(int),
                     round=rec["round"].astype(int))[["read_id", "agent", "post_id", "round"]]
    authors = posts[[POST_ID, POST_AUTHOR]].rename(columns={POST_ID: "post_id", POST_AUTHOR: "author"})
    rec = rec.merge(authors, on="post_id", how="inner")
    rec = rec[rec["agent"] != rec["author"]]
    return attach_follow_status(rec, follow)


def followed_timeline_share(recommendations, posts, follow):
    """Mean share of a timeline that comes from followed accounts, averaged
    over all reads. NaN when the run has no recommendations table."""
    reads = timeline_reads(recommendations, posts, follow)
    if reads.empty:
        return float("nan")
    return float(reads.groupby("read_id")["followed"].mean().mean())


def attach_interaction(exp, events):
    """Adds `interacted`: did the agent react to or reply to that post in
    the exposure round or later?"""
    ev = events[["actor", "post_id", "round"]].dropna().astype(int)
    first_hit = ev.groupby(["actor", "post_id"], as_index=False)["round"].min() \
        .rename(columns={"actor": "agent", "round": "first_interaction_round"})
    out = exp.merge(first_hit, on=["agent", "post_id"], how="left")
    out["interacted"] = out["first_interaction_round"].notna() & \
        (out["first_interaction_round"] >= out["round"])
    return out.drop(columns=["first_interaction_round"])

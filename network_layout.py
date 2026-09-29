"""
network_layout.py — static network figures shared by the scripts and the dashboard
==================================================================================

No Streamlit dependency, so analysis scripts can import it directly.

Layout: two-level, community-grouped
    1. lay out a community-level graph (one node per community, edge weight
       = ties between communities), so strongly tied communities sit together
    2. give each community a circular region sized by sqrt(size), remove
       overlaps and large gaps
    3. lay out each community's own subgraph inside its region, with
       Kamada-Kawai (even spacing); the spring layout is kept for the
       community-level graph in step 1, where Kamada-Kawai would overlap
       disconnected parts

Colors: fixed categorical palettes in C0, C1, ... order (C0 = largest
community), never cycled. community_color(): the validated palette, with
communities past its end sharing one neutral gray (thesis figures).
distinct_color(): a separate color for every community (dashboard). Both
palettes were validated for colorblind separation and contrast against
their surfaces.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "edge": "#898781", "other": "#c3c2b7",
        "palette": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                    "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    # Dark theme = the dashboard's design tokens. Six community colors of
    # similar perceived brightness (OKLCH L 0.61-0.67), validated on #151821:
    # lightness band, chroma, adjacent CVD and normal-vision separation, 3:1
    # contrast. Communities beyond the sixth share the neutral gray.
    "dark": {
        "surface": "#151821", "text": "#F1F3F7", "text2": "#9BA3B0", "muted": "#6F7785",
        "grid": "#2A2F3A", "edge": "#9BA3B0", "other": "#565D6B",
        "palette": ["#4D8FE8", "#DF7033", "#35A978", "#BE8900", "#D15C8F", "#8A6FD1"],
        # One color per community in the dashboard: the six above first, then
        # 18 more picked greedily to be as far as possible (OKLab) from every
        # color already in the list, from the accent purple and from the
        # gray/text tones, all with >= 3:1 contrast on the surface. Past about
        # ten colors some pairs inevitably look alike, so the interactive
        # views back color up with hover labels and legend filtering.
        "extended": ["#4D8FE8", "#DF7033", "#35A978", "#BE8900", "#D15C8F", "#8A6FD1",
                     "#FD95DC", "#B2CB52", "#47D6CF", "#EEAD89", "#6C7E1E", "#A1614B",
                     "#1C8580", "#BCB4F4", "#FB8083", "#5CB4EF", "#92618E", "#5575A9",
                     "#B8AC68", "#C88EF1", "#4399B2", "#5AC576", "#BD7675", "#859454"],
    },
}


def community_color(cid, theme="light"):
    pal = THEMES[theme]["palette"]
    return pal[cid] if cid is not None and 0 <= cid < len(pal) else THEMES[theme]["other"]


def distinct_color(cid, theme="dark"):
    """A separate color for every community id (no shared gray bucket).
    Uses the theme's extended list, then golden-angle hues for very large
    community counts, so a community keeps its color in every view."""
    ext = THEMES[theme].get("extended", THEMES[theme]["palette"])
    if cid is None or cid < 0:
        return THEMES[theme]["other"]
    if cid < len(ext):
        return ext[cid]
    hue = (cid * 137.508) % 360
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((hue / 360, 0.55, 0.85)))


def sort_communities(communities):
    """Largest first (ties broken by smallest agent id): C0, C1, ... stay
    stable across every view."""
    return sorted((set(c) for c in communities), key=lambda c: (-len(c), min(c)))


def louvain(H, weight="weight", seed=42):
    if H.number_of_edges() == 0:
        return [{n} for n in H.nodes()]
    return sort_communities(nx.algorithms.community.louvain_communities(H, weight=weight, seed=seed))


def _spring(H, seed=42):
    n = H.number_of_nodes()
    k = max(0.08, 2.2 / np.sqrt(max(n, 1)))
    return nx.spring_layout(H, seed=seed, k=k, iterations=250, weight="weight")


def _inner_layout(H, seed=42):
    """Layout of one community inside its region.

    Kamada-Kawai (unweighted: it places nodes by graph distance) spreads a
    community evenly over its disc; a spring layout pulls chain-like
    communities - common in a seeded small-world follow network - into thin
    strings where nodes overlap. KK needs a connected graph, so a
    disconnected community (rare with Louvain) falls back to the spring
    layout. Deterministic: KK starts from a circle, the spring layout is
    seeded."""
    if H.number_of_nodes() >= 3 and nx.is_connected(H):
        return nx.kamada_kawai_layout(H, weight=None)
    return _spring(H, seed)


def _pack_circles(centers, radii, gap, iterations=400):
    c = np.array(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    min_dist = r[:, None] + r[None, :] + gap
    for _ in range(iterations):
        c -= 0.02 * (c - c.mean(axis=0))
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        overlap = np.clip(min_dist - dist, 0, None)
        np.fill_diagonal(overlap, 0)
        if not overlap.any():
            continue
        direction = diff / np.where(dist[..., None] == np.inf, 1, dist[..., None] + 1e-9)
        c += 0.5 * (overlap[..., None] * direction).sum(axis=1)
    return c


def community_grouped_layout(G, communities, seed=42):
    """Returns (positions, community centers, community radii), all
    normalised into [-1, 1]."""
    H = G.to_undirected()
    groups = [set(c) & set(H.nodes()) for c in communities]
    groups = [g for g in groups if g]
    covered = set().union(*groups) if groups else set()
    groups += [{n} for n in H.nodes() if n not in covered]
    if len(groups) <= 1:
        return _spring(H, seed), np.zeros((1, 2)), np.ones(1)

    member_of = {n: i for i, g in enumerate(groups) for n in g}
    M = nx.Graph()
    M.add_nodes_from(range(len(groups)))
    for u, v, d in H.edges(data=True):
        a, b = member_of[u], member_of[v]
        if a != b:
            w = d.get("weight", 1)
            M.add_edge(a, b, weight=M[a][b]["weight"] + w if M.has_edge(a, b) else w)
    meta = nx.spring_layout(M, seed=seed, weight="weight", iterations=300)
    centers = np.array([meta[i] for i in range(len(groups))])
    sizes = np.array([len(g) for g in groups], dtype=float)
    radii = np.maximum(0.9 * np.sqrt(sizes / sizes.sum()), 0.02)
    centers = _pack_circles(centers, radii, gap=0.25 * radii.mean())

    pos = {}
    for i, g in enumerate(groups):
        if len(g) == 1:
            pos[next(iter(g))] = centers[i]
            continue
        sub = _inner_layout(H.subgraph(g), seed)
        pts = np.array(list(sub.values()))
        pts -= pts.mean(axis=0)
        pts = pts / (np.abs(pts).max() or 1.0) * radii[i] * 0.85
        for node, p in zip(sub.keys(), pts):
            pos[node] = centers[i] + p

    allp = np.array(list(pos.values()))
    mid = (allp.max(axis=0) + allp.min(axis=0)) / 2
    span = (allp.max(axis=0) - allp.min(axis=0)).max() / 2 or 1.0
    return ({n: (p - mid) / span for n, p in pos.items()}, (centers - mid) / span, radii / span)


def draw_network(ax, G, pos, community_of, theme="light", node_scale=1.0,
                 ghost_nodes=None, size_by=None, labels=None):
    """Draw a directed graph on fixed positions.

    community_of: node -> community id (colors).
    ghost_nodes: nodes drawn as faint dots only (e.g. agents that become
      active later), so the canvas keeps the same frame across snapshots.
    size_by: node -> value for node size (default: degree in G).
    labels: nodes to label with their id."""
    th = THEMES[theme]
    ax.set_facecolor(th["surface"])
    if ghost_nodes:
        gx = [pos[n][0] for n in ghost_nodes if n in pos]
        gy = [pos[n][1] for n in ghost_nodes if n in pos]
        ax.scatter(gx, gy, s=6 * node_scale, c=th["grid"], linewidths=0, zorder=1)

    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    wmax = max(weights) if weights else 1
    intra, inter, iw, xw, ic = [], [], [], [], []
    for (u, v, d) in G.edges(data=True):
        w = 0.4 + 1.4 * d.get("weight", 1) / wmax
        cu, cv = community_of.get(u), community_of.get(v)
        if cu is not None and cu == cv:
            intra.append((u, v)); iw.append(w); ic.append(community_color(cu, theme))
        else:
            inter.append((u, v)); xw.append(w)
    if inter:
        nx.draw_networkx_edges(G, pos, edgelist=inter, ax=ax, width=xw, alpha=0.25,
                               edge_color=th["edge"], arrows=False)
    if intra:
        nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax, width=iw, alpha=0.55,
                               edge_color=ic, arrows=False)

    nodes = [n for n in G.nodes() if G.degree(n) > 0]
    if nodes:
        val = size_by or dict(G.degree(nodes))
        vmax = max(val.get(n, 0) for n in nodes) or 1
        sizes = [(14 + 160 * (val.get(n, 0) / vmax) ** 0.8) * node_scale for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=sizes,
                               node_color=[community_color(community_of.get(n), theme) for n in nodes],
                               edgecolors=th["surface"], linewidths=0.6)
    if labels:
        # to the upper right of the node, with a halo in the surface color so
        # the id stays readable on top of edges and neighbouring nodes
        halo = [patheffects.withStroke(linewidth=2.2, foreground=th["surface"])]
        for n in labels:
            if n in pos:
                ax.annotate(str(n), pos[n], xytext=(4, 4), textcoords="offset points",
                            fontsize=6.5, color=th["text"], weight="bold", zorder=5,
                            path_effects=halo)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


def legend_handles(communities, theme="light", max_colored=None):
    if max_colored is None:
        max_colored = len(THEMES[theme]["palette"])
    handles = [Patch(facecolor=community_color(i, theme), edgecolor="none",
                     label=f"C{i} · {len(c)} agents")
               for i, c in enumerate(communities[:max_colored])]
    rest = communities[max_colored:]
    if rest:
        n = sum(len(c) for c in rest)
        span = f"C{max_colored}" if len(rest) == 1 else f"C{max_colored}–C{len(communities) - 1}"
        handles.append(Patch(facecolor=THEMES[theme]["other"], edgecolor="none",
                             label=f"{span} · {n} agents"))
    return handles


def apply_theme(theme="light"):
    th = THEMES[theme]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5,
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"],
        "savefig.facecolor": th["surface"], "text.color": th["text"],
        "axes.labelcolor": th["text2"], "axes.edgecolor": th["grid"],
        "xtick.color": th["muted"], "ytick.color": th["muted"],
        "axes.titlecolor": th["text"],
    })
    return th


__all__ = ["THEMES", "community_color", "sort_communities", "louvain", "community_grouped_layout",
           "draw_network", "legend_handles", "apply_theme", "plt"]

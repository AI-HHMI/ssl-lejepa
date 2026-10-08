"""Small matplotlib/HTML helpers: each chart returns an inline SVG string, each table an HTML string.
Started from lmd-catalog's scripts/analysis/plots.py (9b1475d); diverges freely. Line and scatter legends sit outside
the axes, on the right (LEGEND_RIGHT), so they never cover data."""

import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Okabe-Ito colorblind-safe palette; grey is reserved for "Other".
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
GREY = "#999999"
MAX_SLICES = 8
LEGEND_RIGHT = dict(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))  # to_svg's tight bbox grows to fit

plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "lmd"})  # fixed salt: same data -> byte-identical SVG


def to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def top_n(counts: dict, n: int = MAX_SLICES) -> dict:
    """Keep the n largest entries and fold the rest into 'Other'."""
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    out = dict(items[:n])
    rest = sum(v for _, v in items[n:])
    if rest:
        out["Other"] = rest
    return out


def pie(counts: dict) -> str:
    counts = top_n(counts)
    colors = [GREY if k == "Other" else COLORS[i] for i, k in enumerate(counts)]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.pie(list(counts.values()), colors=colors, startangle=90, counterclock=False,
           autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", pctdistance=0.78, wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax.legend([f"{k} ({v:,.0f})" if float(v).is_integer() else f"{k} ({v:.3g})" for k, v in counts.items()],
              loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    return to_svg(fig)


def bar(counts: dict, xlabel: str, log: bool = False) -> str:
    """Horizontal bars, largest on top."""
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(items) + 0.8))
    ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS[0])
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    return to_svg(fig)


def hist(values, xlabel: str, bins: int = 30) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(values, bins=bins, color=COLORS[0], edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("volumes")
    return to_svg(fig)


def scatter(groups: dict, xlabel: str, ylabel: str) -> str:
    """groups: label -> (xs, ys)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (label, (xs, ys)), c in zip(groups.items(), COLORS):
        ax.scatter(xs, ys, s=14, alpha=0.6, color=c, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(**LEGEND_RIGHT)
    return to_svg(fig)


def group_bar(rows: list, xlabel: str) -> str:
    """Horizontal bars coloured by group. rows: (label, value, group), drawn top to bottom in the order given."""
    groups = list(dict.fromkeys(g for _, _, g in rows))
    color = {g: COLORS[i] for i, g in enumerate(groups)}
    ys = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(rows) + 0.9))
    ax.barh(ys, [v for _, v, _ in rows], color=[color[g] for _, _, g in rows])
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_xlabel(xlabel)
    ax.legend([Rectangle((0, 0), 1, 1, color=color[g]) for g in groups], groups, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    return to_svg(fig)


def stacked_bar(rows: list, kinds: list, xlabel: str) -> str:
    """Horizontal stacked bars, top to bottom in the order given. rows: (label, group, {kind: value}). A dotted line
    marks where the group changes, and the group's name sits at the right end of its last row."""
    ys = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(rows) + 1.3))
    left = [0] * len(rows)
    for i, kind in enumerate(kinds):
        values = [r[2].get(kind, 0) for r in rows]
        ax.barh(ys, values, left=left, color=GREY if kind.startswith("Other") else COLORS[i], label=kind)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in rows])
    for i, (_, group, _) in enumerate(rows):
        if i and group != rows[i - 1][1]:
            ax.axhline(ys[i] + 0.5, color="#bbb", linestyle=":", linewidth=0.8)
        if i == len(rows) - 1 or rows[i + 1][1] != group:
            ax.text(1.0, ys[i], group, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color="#777")
    ax.set_xlabel(xlabel)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    return to_svg(fig)


def box(groups: dict, xlabel: str) -> str:
    """Horizontal box plots on a log axis, top to bottom in the order given. groups: label -> values."""
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(groups) + 0.9))
    ax.boxplot(list(groups.values()), orientation="horizontal", patch_artist=True, boxprops={"facecolor": COLORS[0], "alpha": 0.5}, medianprops={"color": "black"}, flierprops={"markersize": 3})
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels(list(groups))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    return to_svg(fig)


def lines(series: dict, ylabel: str, log: bool = False) -> str:
    """Step lines over time. series: label -> (dates, values)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for (label, (xs, ys)), c in zip(series.items(), COLORS):
        ax.step(xs, ys, where="post", color=GREY if label == "Other" else c, label=label, linewidth=1.8)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(**LEGEND_RIGHT)
    fig.autofmt_xdate()
    return to_svg(fig)


def fmt(c) -> str:
    """Table cell text: ints with separators, floats to 2 places -- or scientific notation if that would show a nonzero value as 0.00."""
    if isinstance(c, int):
        return f"{c:,}"
    if isinstance(c, float):
        return f"{c:.2e}" if c != 0 and abs(c) < 0.005 else f"{c:,.2f}"
    return c


def table(header: list, rows: list) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td class='{'n' if not isinstance(c, str) else ''}'>{html.escape(fmt(c))}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"

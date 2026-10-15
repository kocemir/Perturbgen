"""Recreates gene_query_jepa_query.png/.pdf for the winning config
(src90m -> 6h/10h, n_src_queries=64, n_top_common_queries=48), matching the visual
style of the existing docs/qj_architecture figures as closely as practical without
their original source file."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#F3F0E6"
INK = "#1E2430"
SUBINK = "#5B6472"

AMBER_BORDER, AMBER_FILL = "#9A6B00", "#FBF1D4"
BLUE_BORDER, BLUE_FILL = "#1C5C8C", "#E2EFF8"
PURPLE_BORDER, PURPLE_FILL = "#5B3A91", "#EFE6F8"
RED_BORDER, RED_FILL = "#9B2C2C", "#FBE7E7"
GREY_BORDER, GREY_FILL = "#6B7280", "#F0F1F3"
WHITE_BORDER, WHITE_FILL = "#444444", "#FFFFFF"

FIG_W, FIG_H = 20.0, 11.6


def box(ax, x, y, w, h, border, fill, lw=1.8, r=0.018):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=border, facecolor=fill,
        mutation_aspect=1, transform=ax.transAxes, zorder=2,
    ))


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(BG)

    # ---- Title ----
    ax.text(0.03, 0.965, "Query", fontsize=30, fontweight="bold", color=INK, va="top")
    ax.text(0.145, 0.965, "·", fontsize=30, color=SUBINK, va="top")
    ax.text(0.165, 0.965, "src90m $\\rightarrow$ 6h / 10h", fontsize=26, fontweight="bold",
            color="#1C5C8C", va="top")
    ax.text(0.03, 0.918, r"One question is a vector in $\mathbb{R}^{768}$.  Each time "
                          r"$t\in\{1,2\}$ draws its own 128 gene questions, then appends "
                          r"1 cell question.",
            fontsize=14.5, color=SUBINK, va="top")

    # ---- Row 1: 3 boxes ----
    y1, h1 = 0.57, 0.33
    xs = [0.03, 0.355, 0.68]
    ws = [0.305, 0.305, 0.29]

    # Box 1: Which genes
    x, w = xs[0], ws[0]
    box(ax, x, y1, w, h1, AMBER_BORDER, AMBER_FILL)
    cx = x + w / 2
    ax.text(cx, y1 + h1 - 0.035, "1   Which genes", fontsize=17, fontweight="bold",
            color=AMBER_BORDER, ha="center", va="top")
    lines = [
        (r"64 uniform from genes in $x^{\mathrm{src}}$", 13.5, INK),
        (r"48 deterministic, absent from $x^{\mathrm{src}}$", 13.5, INK),
        (r"16 weighted-random, absent from $x^{\mathrm{src}}$", 13.5, INK),
        (r"$w^{\mathrm{top}}(g){=}p(g)^{1/2}{+}0.02$,  arg top-48", 12, AMBER_BORDER),
        (r"$w^{\mathrm{rand}}(g){=}(1{-}p(g))^{1/2}{+}0.02$", 12, AMBER_BORDER),
        (r"$p(g) = \widehat{\mathrm{Pr}}[g\ \mathrm{on} \mid \mathrm{type}, t]$", 12, SUBINK),
        ("this cell's own target list is not used", 11.5, SUBINK),
        ("fresh draws at $t{=}1$ and $t{=}2$", 11.5, SUBINK),
    ]
    yy = y1 + h1 - 0.072
    for text, fs, col in lines:
        ax.text(cx, yy, text, fontsize=fs, color=col, ha="center", va="top")
        yy -= 0.032

    # Box 2: Source rank (unchanged)
    x, w = xs[1], ws[1]
    box(ax, x, y1, w, h1, BLUE_BORDER, BLUE_FILL)
    cx = x + w / 2
    ax.text(cx, y1 + h1 - 0.035, "2   Source rank", fontsize=17, fontweight="bold",
            color=BLUE_BORDER, ha="center", va="top")
    lines = [
        (r"$r_{\mathrm{src}}(g) = $ index of $g$ in $x^{\mathrm{src}}$", 13.5, INK),
        (r"$R[r]\in\mathbb{R}^{768}$, learned", 13.5, INK),
        (r"table $R\in\mathbb{R}^{512\times 768}$", 13, BLUE_BORDER),
        ("rows $0,\\ldots,L_s{-}1$ are real ranks, $L_s{\\leq}512$", 11.5, SUBINK),
        ("$r=-1$ uses the last row, one shared code", 11.5, SUBINK),
        ("unchanged from the base config", 11, SUBINK),
    ]
    yy = y1 + h1 - 0.10
    for text, fs, col in lines:
        ax.text(cx, yy, text, fontsize=fs, color=col, ha="center", va="top")
        yy -= 0.034

    # Box 3: Target time
    x, w = xs[2], ws[2]
    box(ax, x, y1, w, h1, PURPLE_BORDER, PURPLE_FILL)
    cx = x + w / 2
    ax.text(cx, y1 + h1 - 0.035, "3   Target time", fontsize=17, fontweight="bold",
            color=PURPLE_BORDER, ha="center", va="top")
    lines = [
        (r"$T[t]\in\mathbb{R}^{768}$, learned", 13.5, INK),
        (r"table $T\in\mathbb{R}^{3\times 768}$", 13.5, PURPLE_BORDER),
        ("rows 1, 2 are 6h, 10h", 12.5, INK),
        ("90m is the source, not predicted", 11.5, SUBINK),
        ("same vector added to every question", 11.5, SUBINK),
        ("encoder's own time index stays fixed at 1", 11.5, SUBINK),
    ]
    yy = y1 + h1 - 0.10
    for text, fs, col in lines:
        ax.text(cx, yy, text, fontsize=fs, color=col, ha="center", va="top")
        yy -= 0.034

    # ---- Formula bar ----
    yf, hf = 0.455, 0.095
    box(ax, 0.03, yf, 0.94, hf, "#444444", "#FFFFFF", lw=1.6)
    ax.text(0.5, yf + hf - 0.020,
             r"$q \;=\; E[g] \;+\; R[\,r_{\mathrm{src}}(g)\,] \;+\; T[t] \;\in\; \mathbb{R}^{768}$",
             fontsize=21, color=INK, ha="center", va="top")
    ax.text(0.5, yf + 0.016,
             r"$E[g]\in\mathbb{R}^{768}$ is $g$'s row in the encoder token table. "
             r"Context stays in $H_{\mathrm{src}}$, which the predictor reads as memory.",
             fontsize=12.5, color=SUBINK, ha="center", va="bottom")

    # ---- Row 3: 5 boxes ----
    y3, h3 = 0.17, 0.265
    specs = [
        ("In the source", "51 of the 64", AMBER_BORDER, AMBER_FILL,
         [(r"$E[g]$", "token row"), (r"$R[r]$", "its own rank"), (r"$T[t]$", "this time")]),
        ("Mask-CE", "13 of the 64", RED_BORDER, RED_FILL,
         [(r"$m$", "learned, no name"), (r"$R[r]$", "rank stays"), (r"$T[t]$", "time stays")]),
        ("Top-common absent", "48, deterministic", "#1C6B4A", "#E3F4EC",
         [(r"$E[g]$", "name stays"), (r"$R_{\emptyset}$", "the last row"), (r"$T[t]$", "this time")]),
        ("Random absent", "16, weighted", BLUE_BORDER, BLUE_FILL,
         [(r"$E[g]$", "name stays"), (r"$R_{\emptyset}$", "the last row"), (r"$T[t]$", "this time")]),
        ("Cell question", "slot 129", PURPLE_BORDER, PURPLE_FILL,
         [(r"$c$", "learned, no gene"), ("–", "no rank"), (r"$T[t]$", "time only")]),
    ]
    n = len(specs)
    gap = 0.012
    w3 = (0.94 - gap * (n - 1)) / n
    x3 = 0.03
    for title, sub, border, fill, rows in specs:
        box(ax, x3, y3, w3, h3, border, fill)
        cx = x3 + w3 / 2
        ax.text(cx, y3 + h3 - 0.028, title, fontsize=13.5, fontweight="bold",
                color=border, ha="center", va="top")
        ax.text(cx, y3 + h3 - 0.058, sub, fontsize=10.5, color=SUBINK, ha="center", va="top")
        ry = y3 + h3 - 0.095
        rh = 0.046
        rw = w3 - 0.02
        for label, caption in rows:
            box(ax, cx - rw / 2, ry - rh, rw, rh, border, "#FFFFFF", lw=1.1, r=0.01)
            ax.text(cx, ry - rh / 2 + 0.012, label, fontsize=13, color=INK, ha="center", va="center")
            ax.text(cx, ry - rh + 0.010, caption, fontsize=8.7, color=SUBINK, ha="center", va="bottom")
            ry -= rh + 0.014
        x3 += w3 + gap

    # ---- Bottom summary ----
    yb, hb = 0.03, 0.135
    box(ax, 0.03, yb, 0.94, hb, AMBER_BORDER, AMBER_FILL)
    ax.text(0.055, yb + hb - 0.025, "What enters the predictor", fontsize=15.5,
            fontweight="bold", color=AMBER_BORDER, va="top")
    ax.text(0.055, yb + hb - 0.062,
             r"$[\,q_1,\ldots,q_{128},\ q_{\mathrm{cell}}\,] \in \mathbb{R}^{B\times 129\times 768}$",
             fontsize=18, color=INK, va="top")
    ax.text(0.055, yb + 0.018,
             r"$q_{\mathrm{cell}} = c + T[t]$, with $c\in\mathbb{R}^{768}$. "
             r"The 13 mask-CE slots are the questions that train $\mathcal{L}_{\mathrm{mask}}$.",
             fontsize=11.5, color=SUBINK, va="bottom")

    out_base = Path(__file__).resolve().parent / "gene_query_jepa_query_src90m_qsb64tc48"
    fig.savefig(str(out_base) + ".png", dpi=150, facecolor=BG)
    fig.savefig(str(out_base) + ".pdf", facecolor=BG)
    print("wrote", str(out_base) + ".png", "and .pdf")


if __name__ == "__main__":
    main()

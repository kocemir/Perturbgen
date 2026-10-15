"""Recreates gene_query_jepa_loss.png/.pdf for the winning config
(src90m -> 6h/10h, lambda_contrastive=0, 13 mask-CE slots), matching the visual
style of the existing docs/qj_architecture figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#F3F0E6"
INK = "#1E2430"
SUBINK = "#5B6472"

AMBER_BORDER, AMBER_FILL = "#9A6B00", "#FBF1D4"
TEAL_BORDER, TEAL_FILL = "#1C6B4A", "#E3F4EC"
PURPLE_BORDER, PURPLE_FILL = "#5B3A91", "#EFE6F8"
RED_BORDER, RED_FILL = "#9B2C2C", "#FBE7E7"
GREY_BORDER, GREY_FILL = "#8A8F98", "#E8E8E8"
NAVY_FILL, NAVY_BORDER = "#1A1030", "#8B6FC9"

FIG_W, FIG_H = 20.0, 11.2


def box(ax, x, y, w, h, border, fill, lw=1.8, r=0.018, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=border, facecolor=fill, linestyle=ls,
        mutation_aspect=1, transform=ax.transAxes, zorder=2,
    ))


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(BG)

    # ---- Title ----
    ax.text(0.03, 0.965, "Loss", fontsize=30, fontweight="bold", color=INK, va="top")
    ax.text(0.125, 0.965, "·", fontsize=30, color=SUBINK, va="top")
    ax.text(0.145, 0.965, "src90m $\\rightarrow$ 6h / 10h", fontsize=26, fontweight="bold",
            color="#1C5C8C", va="top")
    ax.text(0.03, 0.918,
            r"One number per target time, then the mean over $t\in\{1,2\}$.  "
            r"The cell question has weight 0.",
            fontsize=14.5, color=SUBINK, va="top")

    # ---- Formula bar ----
    yf, hf = 0.80, 0.095
    box(ax, 0.03, yf, 0.94, hf, NAVY_BORDER, NAVY_FILL, lw=1.6)
    ax.text(0.5, yf + hf / 2,
             r"$\mathcal{L} = \mathrm{mean}_t(\mathcal{L}_{\mathrm{var}}"
             r" + 0.25\,\mathcal{L}_{\mathrm{abs}} + 0.1\,\mathcal{L}_{\mathrm{mask}})$",
             fontsize=25, color="#FFFFFF", ha="center", va="center")

    # ---- Row of 4 boxes ----
    y1, h1 = 0.335, 0.435
    gap = 0.015
    w1 = (0.94 - 3 * gap) / 4
    xs = [0.03 + i * (w1 + gap) for i in range(4)]

    specs = [
        (r"$\mathcal{L}_{\mathrm{var}}$", r"$\lambda=1$", AMBER_BORDER, AMBER_FILL,
         r"$\mathrm{mean}_{g\in P}\ \frac{1}{768}\| \hat z_g - z^\star_g \|^2$",
         [r"$P$: gene is on at time $t$", "and the slot is outside the 13 masks",
          r"$z^\star_g$ is its row of $H_{\mathrm{tgt}}$", r"both vectors raw, in $\mathbb{R}^{768}$"],
         False),
        (r"$\mathcal{L}_{\mathrm{abs}}$", r"$\lambda=0.25$", TEAL_BORDER, TEAL_FILL,
         r"$\mathrm{mean}_{g\in A}\ \frac{1}{768}\| \hat z_g - a_{\mathrm{null}} \|^2$",
         [r"$A$: gene is off at time $t$", "and the slot is outside the 13 masks",
          r"$a_{\mathrm{null}}\in\mathbb{R}^{768}$ is fixed", r"$\| a_{\mathrm{null}} \| = \sqrt{768}$"],
         False),
        (r"$\mathcal{L}_{\mathrm{NCE}}$", r"$\lambda=0$  (dropped)", GREY_BORDER, GREY_FILL,
         r"$\mathrm{CE}(\mathrm{softmax}(S/0.1))$",
         [r"$S_{ij}=\hat u_i \cdot u^\star_j$", "same slots as $\\mathcal{L}_{\\mathrm{var}}$, over the batch",
          "no longer computed in this run", "kept here only for comparison"],
         True),
        (r"$\mathcal{L}_{\mathrm{mask}}$", r"$\lambda=0.1$", RED_BORDER, RED_FILL,
         r"$\mathrm{CE}(W\hat z/\| \hat z\|,\ g)$",
         [r"the 13 mask-CE slots", r"$W:\mathbb{R}^{768}\to\mathbb{R}^{V}$",
          r"$V$ is the local gene vocabulary", "label is the hidden gene id"],
         False),
    ]

    for (title, lam, border, fill, formula, lines, dropped), x in zip(specs, xs):
        ls = "dashed" if dropped else "solid"
        box(ax, x, y1, w1, h1, border, fill, ls=ls)
        cx = x + w1 / 2
        ax.text(x + 0.018, y1 + h1 - 0.045, title, fontsize=19, color=border,
                ha="left", va="top")
        ax.text(x + w1 - 0.018, y1 + h1 - 0.045, lam, fontsize=14, color=border,
                ha="right", va="top", style=("italic" if dropped else "normal"))

        fbh = 0.095
        fby = y1 + h1 - 0.10 - fbh
        box(ax, x + 0.02, fby, w1 - 0.04, fbh, border, "#FFFFFF", lw=1.3, r=0.012)
        ax.text(cx, fby + fbh / 2, formula, fontsize=13.5, color=INK, ha="center", va="center")

        yy = fby - 0.045
        for i, line in enumerate(lines):
            col = SUBINK if i else INK
            ax.text(cx, yy, line, fontsize=12, color=col, ha="center", va="top")
            yy -= 0.052

    # ---- Bottom legend ----
    yb, hb = 0.03, 0.27
    box(ax, 0.03, yb, 0.94, hb, "#444444", "#FFFFFF", lw=1.6)
    ax.text(0.055, yb + hb - 0.035, "Where the 128 gene questions go", fontsize=17,
            fontweight="bold", color=INK, va="top")
    rows = [
        (r"On at time $t$", r"$\rightarrow$", r"$\mathcal{L}_{\mathrm{var}}$ only  (not $\mathcal{L}_{\mathrm{NCE}}$ — dropped)"),
        (r"Off at time $t$", r"$\rightarrow$", r"$\mathcal{L}_{\mathrm{abs}}$."),
        ("The 13 mask slots", r"$\rightarrow$", r"$\mathcal{L}_{\mathrm{mask}}$ alone."),
    ]
    yy = yb + hb - 0.085
    for a, arrow, b in rows:
        ax.text(0.055, yy, a, fontsize=14, color=INK, va="top")
        ax.text(0.26, yy, arrow, fontsize=14, color=SUBINK, va="top")
        ax.text(0.285, yy, b, fontsize=14, color=INK, va="top")
        yy -= 0.052

    ax.text(0.52, yb + hb - 0.035,
            "On and off are read from the target, after the quiz is drawn.",
            fontsize=12.5, color=SUBINK, va="top", ha="left")
    ax.text(0.52, yb + hb - 0.065,
            "Genes come from §2's three pools: source, top-common, random.",
            fontsize=12.5, color=SUBINK, va="top", ha="left")

    badge_specs = [
        ("on at $t$", AMBER_BORDER, AMBER_FILL),
        ("off at $t$", TEAL_BORDER, TEAL_FILL),
        ("13 masks", RED_BORDER, RED_FILL),
    ]
    bx = 0.72
    bw, bh = 0.075, 0.075
    for label, border, fill in badge_specs:
        box(ax, bx, yb + 0.045, bw, bh, border, fill, lw=1.4, r=0.014)
        ax.text(bx + bw / 2, yb + 0.045 + bh / 2, label, fontsize=12.5,
                fontweight="bold", color=border, ha="center", va="center")
        bx += bw + 0.02

    out_base = Path(__file__).resolve().parent / "gene_query_jepa_loss_src90m_qsb64tc48"
    fig.savefig(str(out_base) + ".png", dpi=150, facecolor=BG)
    fig.savefig(str(out_base) + ".pdf", facecolor=BG)
    print("wrote", str(out_base) + ".png", "and .pdf")


if __name__ == "__main__":
    main()

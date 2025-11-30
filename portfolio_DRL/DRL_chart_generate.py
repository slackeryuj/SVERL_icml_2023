import matplotlib
matplotlib.use("Agg")  # headless backend: only saves PNGs

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def add_box(ax, x, y, w, h, text,
            fc="#FFFFFF", ec="#333333", tc="#111111",
            fontsize=9):
    """Rounded rectangle with centered multi-line text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        path_effects=[pe.withStroke(linewidth=0.4, foreground="#000000", alpha=0.2)],
    )
    return box


def add_arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.4):
    """Straight arrow from (x1,y1) to (x2,y2)."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)


# -------------------------------------------------------------------
# 1. Overall flow: Stage 1 -> Stage 2 -> Outputs
# -------------------------------------------------------------------
def draw_overall_flow(filename="overall_flow_paper.png"):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_title("Overall signal-to-policy flow", fontsize=14, pad=14)

    base_x = 0.10
    w = 0.22
    gap = 0.06
    h = 0.30
    y = 0.40

    add_box(
        ax, base_x, y, w, h,
        "Stage 1\nXGBoost signal layer\n\n"
        "• technical indicators\n"
        "• Fama–French factors\n"
        "• macro variables",
        fc="#E3F2FD", ec="#1565C0", tc="#0D47A1", fontsize=9,
    )

    add_box(
        ax, base_x + w + gap, y, w, h,
        "Stage 2\nPPO DRL policy layer\n\n"
        "• state $S_t$ from Stage 1\n"
        "• actor–critic MLPs\n"
        "• mean–CVaR reward",
        fc="#E8F5E9", ec="#2E7D32", tc="#1B5E20", fontsize=9,
    )

    add_box(
        ax, base_x + 2 * (w + gap), y, w, h,
        "Outputs (portfolio level)\n\n"
        "• weights $w_t$\n"
        "• wealth / P&L path\n"
        "• CVaR, Sharpe, MaxDD",
        fc="#ECEFF1", ec="#455A64", tc="#263238", fontsize=9,
    )

    mid_y = y + h / 2
    add_arrow(ax, base_x + w, mid_y, base_x + w + gap, mid_y)
    add_arrow(ax, base_x + 2*w + gap, mid_y, base_x + 2*w + 2*gap, mid_y)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.15)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# 2. Stage 1 – XGBoost architecture (per asset)
# -------------------------------------------------------------------
def draw_xgboost_architecture(filename="xgboost_arch_paper.png"):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_title("Stage 1 – XGBoost architecture (per asset)",
                 fontsize=14, pad=14)

    base_x = 0.10
    w = 0.23
    gap = 0.06
    h = 0.32
    y = 0.40

    # Input features
    add_box(
        ax, base_x, y, w, h,
        "Feature vector $X_{i,t}$\n\n"
        "• technical indicators\n"
        "• factor & macro lags",
        fc="#FFF3E0", ec="#FFB300", tc="#E65100", fontsize=10,
    )

    # XGBoost ensemble
    add_box(
        ax, base_x + w + gap, y, w, h,
        "XGBoost ensemble\n\n"
        "$\\hat y_{i,t} = \\sum_{k=1}^K f_{i,k}(X_{i,t})$",
        fc="#E8F5E9", ec="#2E7D32", tc="#1B5E20", fontsize=10,
    )

    # Outputs
    add_box(
        ax, base_x + 2 * (w + gap), y, w, h,
        "Outputs\n\n"
        "• predicted return $\\hat y_{i,t}$\n"
        "• Score$(i,t)$ (percentile)\n"
        "• grouped SHAP",
        fc="#E3F2FD", ec="#1565C0", tc="#0D47A1", fontsize=10,
    )

    mid_y = y + h / 2
    add_arrow(ax, base_x + w, mid_y, base_x + w + gap, mid_y)
    add_arrow(ax, base_x + 2*w + gap, mid_y, base_x + 2*w + 2*gap, mid_y)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.15)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# 3. Stage 2 – PPO actor–critic architecture
# -------------------------------------------------------------------
def draw_ppo_architecture(filename="ppo_arch_paper.png",
                          input_dim=128, hidden1=64, hidden2=64, n_actions=10):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_title("Stage 2 – PPO actor–critic architecture",
                 fontsize=14, pad=14)

    base_x = 0.10
    w = 0.17
    gap = 0.04
    h = 0.18

    # ---------- Actor row ----------
    y_top = 0.70
    add_box(
        ax, base_x, y_top, w, h,
        f"Actor input\nstate $S_t$\n({input_dim} dims)",
        fc="#E3F2FD", ec="#1565C0", tc="#0D47A1", fontsize=9,
    )
    add_box(
        ax, base_x + w + gap, y_top, w, h,
        f"Hidden layer 1\n{hidden1} units\nReLU",
        fc="#EDE7F6", ec="#5E35B1", tc="#311B92", fontsize=9,
    )
    add_box(
        ax, base_x + 2*(w + gap), y_top, w, h,
        f"Hidden layer 2\n{hidden2} units\nReLU",
        fc="#EDE7F6", ec="#5E35B1", tc="#311B92", fontsize=9,
    )
    add_box(
        ax, base_x + 3*(w + gap), y_top, w, h,
        f"Actor output\npolicy mean/std\n({n_actions} actions)",
        fc="#E8F5E9", ec="#2E7D32", tc="#1B5E20", fontsize=9,
    )

    mid_top = y_top + h / 2
    add_arrow(ax, base_x + w, mid_top, base_x + w + gap, mid_top)
    add_arrow(ax, base_x + 2*w + gap, mid_top, base_x + 2*w + 2*gap, mid_top)
    add_arrow(ax, base_x + 3*w + 2*gap, mid_top, base_x + 3*w + 3*gap, mid_top)

    ax.text(0.10, y_top + h + 0.03, "Actor network",
            fontsize=9, color="#555555", ha="left")

    # ---------- Critic row ----------
    y_bot = 0.35
    add_box(
        ax, base_x, y_bot, w, h,
        f"Critic input\nstate $S_t$\n({input_dim} dims)",
        fc="#E0F2F1", ec="#00897B", tc="#004D40", fontsize=9,
    )
    add_box(
        ax, base_x + w + gap, y_bot, w, h,
        f"Hidden layer 1\n{hidden1} units\nReLU",
        fc="#E0F2F1", ec="#00897B", tc="#004D40", fontsize=9,
    )
    add_box(
        ax, base_x + 2*(w + gap), y_bot, w, h,
        f"Hidden layer 2\n{hidden2} units\nReLU",
        fc="#E0F2F1", ec="#00897B", tc="#004D40", fontsize=9,
    )
    add_box(
        ax, base_x + 3*(w + gap), y_bot, w, h,
        "Critic output\n$V(S_t)$ (scalar)",
        fc="#ECEFF1", ec="#546E7A", tc="#263238", fontsize=9,
    )

    mid_bot = y_bot + h / 2
    add_arrow(ax, base_x + w, mid_bot, base_x + w + gap, mid_bot)
    add_arrow(ax, base_x + 2*w + gap, mid_bot, base_x + 2*w + 2*gap, mid_bot)
    add_arrow(ax, base_x + 3*w + 2*gap, mid_bot, base_x + 3*w + 3*gap, mid_bot)

    ax.text(0.10, y_bot + h + 0.03, "Critic network",
            fontsize=9, color="#555555", ha="left")

    ax.text(
        0.5, 0.12,
        "Actor and critic share the same state $S_t$ but use separate two-layer MLPs\n"
        "trained jointly with the PPO objective.",
        ha="center", va="center", fontsize=8.5, color="#555555",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.13)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# run everything
# -------------------------------------------------------------------
if __name__ == "__main__":
    draw_overall_flow()
    draw_xgboost_architecture()
    draw_ppo_architecture()
    print("Saved: overall_flow_paper.png, xgboost_arch_paper.png, ppo_arch_paper.png")

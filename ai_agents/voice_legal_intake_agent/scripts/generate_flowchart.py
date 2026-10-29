"""Generate the repository's flat RGB architecture diagram."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BACKGROUND = "#EEF2FF"
BORDER = "#4F46E5"
TEXT = "#1E293B"


def box(ax, x: float, y: float, label: str, width: float = 1.45) -> None:
    patch = FancyBboxPatch(
        (x, y), width, 0.62,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        edgecolor=BORDER,
        facecolor="white",
        linewidth=2.2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.31, label, ha="center", va="center", color=TEXT, fontsize=10.5, weight="bold", wrap=True)


def main() -> None:
    figure, axis = plt.subplots(figsize=(13, 2.6), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    labels = ["Voice input", "Krisp STT", "LangGraph\nintake agent", "Human-reviewed\nsummary", "Local SQLite\nrecord"]
    positions = [0.25, 2.18, 4.11, 6.04, 7.97]
    for position, label in zip(positions, labels):
        box(axis, position, 0.37, label)
    for position in positions[:-1]:
        axis.annotate("", xy=(position + 1.88, 0.68), xytext=(position + 1.52, 0.68), arrowprops={"arrowstyle": "->", "lw": 2.5, "color": BORDER, "mutation_scale": 22})
    box(axis, 3.93, 1.17, "Text input", width=1.8)
    axis.annotate("", xy=(4.84, 1.0), xytext=(4.84, 1.16), arrowprops={"arrowstyle": "->", "lw": 2.5, "color": BORDER, "mutation_scale": 22})
    axis.set_xlim(0, 9.75)
    axis.set_ylim(0, 1.95)
    axis.axis("off")
    figure.savefig("assets/how_it_works.png", dpi=150, pad_inches=0.25, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    main()

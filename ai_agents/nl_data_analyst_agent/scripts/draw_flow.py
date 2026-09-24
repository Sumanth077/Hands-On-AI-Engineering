"""Regenerate the project's horizontal architecture PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw() -> None:
    fig, axis = plt.subplots(figsize=(16, 3.0))
    fig.patch.set_facecolor("#EEF2FF")
    axis.set_facecolor("#EEF2FF")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 3.0)
    axis.axis("off")
    labels = [
        "Gradio\nquestion",
        "Eve +\nAI Gateway",
        "Eve SQL tool\n+ approval",
        "SQLite\nread-only",
        "AI Gateway\nexplains",
        "Answer +\nSQL + data",
    ]
    positions = [0.3, 2.95, 5.6, 8.25, 10.9, 13.55]
    for x, label in zip(positions, labels):
        box = FancyBboxPatch((x, 1.0), 2.2, 1.3, boxstyle="round,pad=0.12,rounding_size=0.18",
                             linewidth=2, edgecolor="#4F46E5", facecolor="#FFFFFF")
        axis.add_patch(box)
        axis.text(x + 1.1, 1.65, label, ha="center", va="center", fontsize=11.5,
                  fontweight="bold", color="#1E293B")
    for x in positions[:-1]:
        arrow = FancyArrowPatch((x + 2.35, 1.65), (x + 2.58, 1.65), arrowstyle="-|>",
                                mutation_scale=22, linewidth=2.5, color="#4F46E5")
        axis.add_patch(arrow)
    target = Path(__file__).resolve().parents[1] / "assets" / "how_it_works.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="#EEF2FF")
    plt.close(fig)
    from PIL import Image

    with Image.open(target) as image:
        image.convert("RGB").save(target)


if __name__ == "__main__":
    draw()

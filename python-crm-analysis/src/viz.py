"""Единый стиль графиков проекта и сохранение в reports/figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Валидированная CVD-безопасная категориальная палитра (фиксированный порядок слотов).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY, ACCENT, GOOD = PALETTE[0], PALETTE[1], PALETTE[2]
MUTED = "#a8a69c"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e8e7e2"
SEQ_BLUES = "Blues"


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
    plt.rcParams.update({
        "figure.figsize": (10, 5), "figure.dpi": 110, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": MUTED, "grid.color": GRID, "text.color": TEXT,
        "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)


def pct_axis(ax, axis="y", decimals=0):
    fmt = mtick.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def eur_axis(ax, axis="y"):
    fmt = mtick.FuncFormatter(lambda v, _: f"€{v/1000:,.0f}k" if abs(v) >= 1000 else f"€{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

from __future__ import annotations

import csv
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/research/results/csa_top_k_20260524/all_runs_summary.csv"
REMOTE_ENV = ROOT / "docs/research/results/csa_top_k_20260524/remote_environment.txt"
OUT_DIR = ROOT / "docs/research/reports"
OUT_PDF = OUT_DIR / "csa_pilot_report_20260524.pdf"

PAGE_W = 8.5
PAGE_H = 11.0
BLUE = "#1F3A5F"
INK = "#1D2433"
MUTED = "#64748B"
ORANGE = "#F59E0B"
TEAL = "#0F766E"
RED = "#B91C1C"
LIGHT = "#F4F7FB"
GRID = "#D8DEE9"


def evidence_commit() -> str:
    try:
        for line in REMOTE_ENV.read_text().splitlines():
            if line.startswith("repo_commit="):
                return line.split("=", 1)[1][:12]
    except Exception:
        pass
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_rows() -> list[dict]:
    with RESULTS.open() as f:
        return list(csv.DictReader(f))


def by_run_root(rows: list[dict], run_root: str) -> list[dict]:
    selected = [row for row in rows if row["run_root"] == run_root]
    order = {"dense": 0, "csa-k1": 1, "csa-k2": 2, "csa-k4": 3, "csa-k8": 4, "csa-k16": 5}
    return sorted(selected, key=lambda row: order.get(row["run_name"], 99))


def f(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def setup_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")
    return fig, ax


def footer(ax, page: int):
    ax.text(0.65, 0.35, "CSA Pilot Report | 2026-05-24", fontsize=8, color=MUTED)
    ax.text(PAGE_W - 0.65, 0.35, str(page), fontsize=8, color=MUTED, ha="right")


def draw_wrapped(ax, text: str, x: float, y: float, width_chars: int, size: int = 10, color: str = INK, line_gap: float = 0.18, weight: str = "normal"):
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width_chars))
    for i, line in enumerate(lines):
        ax.text(x, y - i * line_gap, line, fontsize=size, color=color, va="top", weight=weight)
    return y - max(len(lines), 1) * line_gap


def section_title(ax, title: str, y: float):
    ax.text(0.65, y, title.upper(), fontsize=10, color=ORANGE, weight="bold", va="top")
    ax.plot([0.65, 7.85], [y - 0.12, y - 0.12], color=GRID, lw=1)
    return y - 0.35


def callout(ax, title: str, body: str, x: float, y: float, w: float, h: float, color: str = BLUE):
    ax.add_patch(plt.Rectangle((x, y - h), w, h, facecolor=LIGHT, edgecolor=color, lw=1.2))
    ax.add_patch(plt.Rectangle((x, y - h), 0.08, h, facecolor=color, edgecolor=color, lw=0))
    ax.text(x + 0.25, y - 0.25, title, fontsize=11, color=color, weight="bold", va="top")
    draw_wrapped(ax, body, x + 0.25, y - 0.62, int(w * 12), size=9, color=INK, line_gap=0.18)


def draw_table(ax, headers: list[str], rows: list[list[str]], x: float, y: float, col_widths: list[float], row_h: float = 0.38, header_color: str = BLUE):
    total_w = sum(col_widths)
    ax.add_patch(plt.Rectangle((x, y - row_h), total_w, row_h, facecolor=header_color, edgecolor=header_color, lw=0))
    cursor = x
    for header, width in zip(headers, col_widths):
        ax.text(cursor + 0.06, y - row_h / 2, header, color="white", fontsize=8, weight="bold", va="center")
        cursor += width
    for ridx, row in enumerate(rows):
        yy = y - row_h * (ridx + 2)
        bg = "#FFFFFF" if ridx % 2 == 0 else "#F8FAFC"
        ax.add_patch(plt.Rectangle((x, yy), total_w, row_h, facecolor=bg, edgecolor=GRID, lw=0.6))
        cursor = x
        for cell, width in zip(row, col_widths):
            ax.text(cursor + 0.06, yy + row_h / 2, cell, color=INK, fontsize=8, va="center")
            cursor += width
    ax.add_patch(plt.Rectangle((x, y - row_h * (len(rows) + 1)), total_w, row_h * (len(rows) + 1), facecolor="none", edgecolor=GRID, lw=0.8))


def draw_bar(ax, labels, values, title, ylabel, color, y_min=None):
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)


def page_cover(pdf: PdfPages, commit: str):
    fig, ax = setup_page()
    ax.add_patch(plt.Rectangle((0, 0), PAGE_W, PAGE_H, facecolor="#FFFFFF", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0, 9.05), PAGE_W, 1.95, facecolor=BLUE, edgecolor="none"))
    ax.text(0.65, 10.45, "CSA PILOT REPORT", fontsize=12, color="#DDEAFE", weight="bold", va="top")
    ax.text(0.65, 10.0, "Can compressed attention compete", fontsize=22, color="white", weight="bold", va="top")
    ax.text(0.65, 9.52, "under equal GPU time?", fontsize=22, color="white", weight="bold", va="top")
    ax.text(0.65, 8.78, "DeepSeek-style Compressed Sparse Attention, small-scale repo pilot", fontsize=11, color=MUTED, va="top")
    callout(
        ax,
        "Headline",
        "Under equal GPU time, dense attention clearly beat this plain-PyTorch CSA implementation. CSA looked competitive only in the less fair fixed-token view.",
        0.65,
        8.55,
        7.2,
        1.35,
        color=RED,
    )
    callout(
        ax,
        "What this report is",
        "A compact research artifact for teaching: one honest result, the fairness rule, the table, the chart, the caveats, and the next clean experiment.",
        0.65,
        6.85,
        7.2,
        1.25,
        color=TEAL,
    )
    y = section_title(ax, "Metadata", 4.95)
    meta_rows = [
        ["Date", "2026-05-24"],
        ["Hardware", "1x NVIDIA RTX 3090, 24GB VRAM"],
        ["Evidence run commit", commit],
        ["Primary evidence folder", "docs/research/results/csa_top_k_20260524"],
    ]
    draw_table(ax, ["Field", "Value"], meta_rows, 0.65, y, [2.35, 4.85], row_h=0.38)
    footer(ax, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_method(pdf: PdfPages):
    fig, ax = setup_page()
    y = section_title(ax, "Research Question", 10.35)
    y = draw_wrapped(
        ax,
        "If CSA can read compressed summaries of old tokens, does increasing top_k recover quality under a fixed GPU-time budget?",
        0.65,
        y,
        92,
        size=13,
        color=INK,
        line_gap=0.25,
        weight="bold",
    )
    y -= 0.25
    callout(
        ax,
        "Fairness rule",
        "Same model, data, batch size, sequence length, optimizer, seed, hardware, and active training-time budget. VRAM is measured as an outcome, not reused to increase batch size.",
        0.65,
        y,
        7.2,
        1.2,
        color=BLUE,
    )
    y -= 1.6
    y = section_title(ax, "Conditions", y)
    rows = [
        ["dense", "Full causal past", "strong baseline"],
        ["csa-k1", "Local window + 1 compressed block", "small compressed memory"],
        ["csa-k2", "Local window + 2 compressed blocks", "slightly more memory"],
        ["csa-k4", "Local window + 4 compressed blocks", "default pilot knob"],
        ["csa-k8", "Local window + 8 compressed blocks", "larger memory"],
        ["csa-k16", "Local window + 16 compressed blocks", "largest memory in sweep"],
    ]
    draw_table(ax, ["Run", "Reads", "Why included"], rows, 0.65, y, [1.2, 3.25, 2.75], row_h=0.42)
    y -= 3.15
    y = section_title(ax, "Important Caveat", y)
    draw_wrapped(
        ax,
        "This is a plain-PyTorch research implementation. Dense attention benefits from optimized PyTorch kernels. Therefore the result is about this implementation and setup, not a universal claim that CSA cannot work.",
        0.65,
        y,
        95,
        size=10,
        color=INK,
        line_gap=0.2,
    )
    footer(ax, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_fixed_time(pdf: PdfPages, fixed_time: list[dict]):
    fig, ax = setup_page()
    y = section_title(ax, "Fixed GPU-Time Result", 10.35)
    draw_wrapped(
        ax,
        "All rows used about 300 active training seconds. This is the headline comparison because each setting receives the same wall-clock compute budget.",
        0.65,
        y,
        95,
        size=10,
        color=INK,
        line_gap=0.2,
    )
    labels = [row["run_name"] for row in fixed_time]
    losses = [f(row, "val_loss") for row in fixed_time]
    chart_ax = fig.add_axes([0.10, 0.52, 0.80, 0.28])
    draw_bar(chart_ax, labels, losses, "Validation loss after equal GPU time", "loss", [BLUE] + [ORANGE] * (len(labels) - 1), y_min=4.0)
    table_rows = [
        [
            row["run_name"],
            f'{f(row, "val_loss"):.4f}',
            f'{int(f(row, "tokens_seen")):,}',
            f'{f(row, "tokens_per_second"):.0f}',
            f'{f(row, "peak_cuda_memory_allocated_gib"):.2f}',
            f'{int(f(row, "raw_token_equivalent_coverage"))}',
        ]
        for row in fixed_time
    ]
    draw_table(
        ax,
        ["Run", "Val loss", "Tokens seen", "Tok/s", "VRAM GiB", "Coverage"],
        table_rows,
        0.65,
        5.2,
        [1.05, 1.0, 1.75, 1.0, 1.0, 1.0],
        row_h=0.38,
    )
    callout(
        ax,
        "Readout",
        "Dense reached 4.3713 validation loss while the CSA rows clustered near 5.43. Increasing top_k did not produce a visible quality recovery.",
        0.65,
        1.95,
        7.2,
        0.95,
        color=RED,
    )
    footer(ax, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_budget(pdf: PdfPages, fixed_time: list[dict]):
    fig, ax = setup_page()
    y = section_title(ax, "Throughput And Memory", 10.35)
    draw_wrapped(
        ax,
        "The fixed-time result is partly explained by tokens processed. Dense trained on more than twice as many tokens in the same active training window.",
        0.65,
        y,
        95,
        size=10,
        color=INK,
        line_gap=0.2,
    )
    labels = [row["run_name"] for row in fixed_time]
    tokens_m = [f(row, "tokens_seen") / 1_000_000 for row in fixed_time]
    vram = [f(row, "peak_cuda_memory_allocated_gib") for row in fixed_time]
    ax1 = fig.add_axes([0.10, 0.56, 0.36, 0.28])
    ax2 = fig.add_axes([0.55, 0.56, 0.36, 0.28])
    draw_bar(ax1, labels, tokens_m, "Tokens seen", "millions", [BLUE] + [ORANGE] * (len(labels) - 1))
    draw_bar(ax2, labels, vram, "Peak allocated VRAM", "GiB", [BLUE] + [ORANGE] * (len(labels) - 1), y_min=5.8)
    y = section_title(ax, "What The Budget Says", 4.8)
    bullets = [
        "Dense processed 33.3M tokens in about 300 seconds.",
        "CSA rows processed about 13.3M to 14.1M tokens in the same time.",
        "CSA allocated slightly more VRAM in this implementation.",
        "Theoretical sparse attention savings did not appear as a practical speed win here.",
    ]
    for bullet in bullets:
        ax.text(0.85, y, f"- {bullet}", fontsize=10, color=INK, va="top")
        y -= 0.38
    callout(
        ax,
        "Why this matters",
        "A same-token comparison can make CSA look competitive, but equal GPU time asks the practical question: what do we get for the same compute budget?",
        0.65,
        2.05,
        7.2,
        1.0,
        color=TEAL,
    )
    footer(ax, 4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_fixed_tokens_and_next(pdf: PdfPages, fixed_tokens: list[dict]):
    fig, ax = setup_page()
    y = section_title(ax, "Fixed-Token Diagnostic", 10.35)
    draw_wrapped(
        ax,
        "The fixed-token view is useful as a diagnostic, but it is not the headline fairness view because different methods can spend different compute per token.",
        0.65,
        y,
        95,
        size=10,
        color=INK,
        line_gap=0.2,
    )
    labels = [row["run_name"] for row in fixed_tokens]
    losses = [f(row, "val_loss") for row in fixed_tokens]
    chart_ax = fig.add_axes([0.10, 0.58, 0.80, 0.24])
    draw_bar(chart_ax, labels, losses, "Validation loss after 204,800 training tokens", "loss", [BLUE] + [ORANGE] * (len(labels) - 1), y_min=6.9)
    y = section_title(ax, "Next Clean Experiment", 5.35)
    draw_wrapped(
        ax,
        "For the teaching video, the simpler experiment is dense vs local-only vs CSA. It asks whether compressed memory helps beyond a local attention window.",
        0.65,
        y,
        95,
        size=10,
        color=INK,
        line_gap=0.2,
    )
    y -= 0.75
    rows = [
        ["dense", "read the full causal past"],
        ["local", "read nearby tokens only"],
        ["csa", "read nearby tokens plus compressed older memory"],
    ]
    draw_table(ax, ["Condition", "Interpretation"], rows, 0.65, y, [1.4, 5.8], row_h=0.42)
    callout(
        ax,
        "Recommended run",
        "3 attention settings x 3 seeds x 10 minutes = about 90 GPU-minutes. If the gap is below 0.03 validation loss, do not claim a winner.",
        0.65,
        2.1,
        7.2,
        1.0,
        color=BLUE,
    )
    footer(ax, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fixed_time = by_run_root(rows, "20260524_055859")
    fixed_tokens = by_run_root(rows, "20260524_055350")
    commit = evidence_commit()

    with PdfPages(OUT_PDF) as pdf:
        pdf.infodict()["Title"] = "CSA Pilot Report"
        pdf.infodict()["Author"] = "vukrosic/deepseek-compressed-sparse-attention"
        pdf.infodict()["Subject"] = "Compressed Sparse Attention pilot results"
        page_cover(pdf, commit)
        page_method(pdf)
        page_fixed_time(pdf, fixed_time)
        page_budget(pdf, fixed_time)
        page_fixed_tokens_and_next(pdf, fixed_tokens)

    print(OUT_PDF)


if __name__ == "__main__":
    main()

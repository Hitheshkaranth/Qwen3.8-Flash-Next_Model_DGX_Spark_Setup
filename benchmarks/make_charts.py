"""Render the README charts from the result JSON files in this folder.

usage: python3 benchmarks/make_charts.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Validated default categorical palette (dataviz skill, references/palette.md), slots 1-3
# (validated all-pairs for CVD in light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def bar_panel(ax, title, labels, values, colors, fmt, unit):
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", length=0)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02, fmt.format(v),
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")


def throughput():
    r = json.load(open(os.path.join(HERE, "throughput_result.json")))["results"]
    o = r["ornith-1.5-35b-a3b (reference, MTP k=1)"]
    q = r["qwen3.8-flash-next (this repo, NVFP4, MTP k=3)"]
    labels = ["Ornith-1.5-35B-A3B\n(reference)", "Qwen3.8-Flash-Next\n(this repo)"]
    colors = [BLUE, ORANGE]
    o_peak, q_peak = o[f"users_{o['peak_at_users']}"], q[f"users_{q['peak_at_users']}"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    bar_panel(axes[0], "Decode speed, 1 user", labels,
              [o["users_1"]["aggregate_tok_s"], q["users_1"]["aggregate_tok_s"]], colors, "{:.1f}", "tokens / s")
    bar_panel(axes[1], "Peak aggregate decode", labels,
              [o_peak["aggregate_tok_s"], q_peak["aggregate_tok_s"]], colors, "{:.1f}", "tokens / s")
    bar_panel(axes[2], "Per-user decode, at peak", labels,
              [o_peak["per_user_tok_s"], q_peak["per_user_tok_s"]], colors, "{:.1f}", "tokens / s")
    fig.suptitle(f"Throughput on one DGX Spark (GB10), 512 output tokens per request. "
                 f"Peak at each server's own --max-num-seqs (Ornith {o['peak_at_users']}, Flash-Next {q['peak_at_users']}).",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "throughput_comparison.png"), dpi=150)
    print("wrote throughput_comparison.png")


def prefill():
    c = json.load(open(os.path.join(HERE, "prefill_result.json")))["chunks"]
    keys = ["1024", "2048", "4096"]
    labels = [f"{k}" + ("\n(shipped)" if c[k].get("chosen") else "") for k in keys]
    colors = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    bar_panel(axes[0], "New 32k prompt: time to first token", labels,
              [c[k]["cold_ttft_s"]["32k"] for k in keys], colors, "{:.1f} s", "seconds (lower is better)")
    bar_panel(axes[1], "Repeat 32k prompt (cache hit)", labels,
              [c[k]["warm_ttft_s"]["32k"] for k in keys], colors, "{:.2f} s", "seconds (lower is better)")
    bar_panel(axes[2], "Stall for streams already running", labels,
              [c[k]["stall_ms_p50_p95"][0] for k in keys], colors, "{:,.0f} ms", "p50 gap between chunks, ms")
    for ax in axes:
        ax.set_xlabel("MAX_NUM_BATCHED_TOKENS (prefill chunk)")
    fig.suptitle("Prefill chunk width: cold speed vs cache reuse vs streaming smoothness (8 concurrent decoders for the stall)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "prefill_chunk_comparison.png"), dpi=150)
    print("wrote prefill_chunk_comparison.png")


throughput()
prefill()

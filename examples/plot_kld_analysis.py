#!/usr/bin/env python3
"""
Plot KLD (Kullback-Leibler Divergence) vs model file size for quantization analysis.

Each model gets a unique color and shape. All models are listed in the legend.

Usage:
    pip install matplotlib
    python examples/plot_kld_analysis.py
"""

import argparse

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    raise SystemExit(1)


# Llama 3.1 8B parameter count (from Hugging Face model card)
LLAMA_31_8B_PARAMS = 8.03e9


def bpw_from_file_size(size_gib: float) -> float:
    """Compute bits-per-weight from file size (GiB) for Llama 3.1 8B."""
    return round((size_gib * (1024**3) * 8) / LLAMA_31_8B_PARAMS, 2)


# --- Edit your data here ---
# Format: (file_size_gib, mean_kld, bpw, label)
# For non-GGUF: use bpw_from_file_size(size_gib) or leave None to auto-compute Updating.
DATA = [
    (30.0, 0.0, 16.0, "Original (bf16)"),
    (5.4, 0.076226, bpw_from_file_size(5.4), "W4A16_GS128"),
    (5.7, 0.048686, bpw_from_file_size(5.7), "W4A16_GS32"),
    (8.6, 0.000899, bpw_from_file_size(8.6), "W8A16_GS128"),
    (8.9, 0.000813, bpw_from_file_size(8.9), "W8A16_GS32"),
    (6.2, 0.033707, bpw_from_file_size(6.2), "FP8_INT4"),
    (5.7, 0.109275, bpw_from_file_size(5.7), "NVFP4"),
    (5.7, 0.089775, bpw_from_file_size(5.7), "NVFP4_New"),
    (8.5, 0.006547, bpw_from_file_size(8.5), "W8A8-FP8_BLOCK"),
    (3.5, 0.1241, 3.50, "IQ3_XS"),
    (3.65, 0.1782, 3.64, "Q3_K_S"),
    (4.0, 0.0744, 4.00, "Q3_K_M"),
    (4.4, 0.0327, 4.42, "IQ4_XS"),
    (4.7, 0.0305, 4.67, "Q4_K_S"),
    (4.9, 0.0267, 4.89, "Q4_K_M"),
    (5.6, 0.0102, 5.57, "Q5_K_S"),
    (5.7, 0.0092, 5.70, "Q5_K_M"),
    (6.6, 0.0040, 6.56, "Q6_K"),
    (8.0, 0.0011, 8.50, "Q8_0"),
]

# Unique markers and colors for each model (cycle if more models than entries)
MARKERS = ["o", "s", "^", "D", "v", "p", "h", "8", "*", "P", "X", "d", "<", ">", "H", "1", "2", "3", "4"]
COLORS = [
    "#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#2e8b57",
    "#6a0dad",
]


def main():
    parser = argparse.ArgumentParser(description="Plot KLD vs file size")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save plot to file instead of displaying",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, (size_gib, kld, bpw, label) in enumerate(DATA):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        legend_label = f"{label} ({kld:.4f}, {bpw:.2f}bpw)"

        ax.scatter(
            size_gib,
            kld,
            color=color,
            s=100,
            marker=marker,
            zorder=3,
            label=legend_label,
        )

    ax.set_xlabel("File Size (GiB)", fontsize=12)
    ax.set_ylabel("Mean KL Divergence (Lower is Better)", fontsize=12)
    ax.set_title(
        "Llama-3.1-8B-Instruct Quantization Analysis: Mean KL Divergence vs. Model File Size",
        fontsize=14,
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        framealpha=0.95,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=-0.02)
    ax.set_xlim(left=2.5, right=32)

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

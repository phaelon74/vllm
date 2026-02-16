#!/usr/bin/env python3
"""
Plot KLD (Kullback-Leibler Divergence) vs model file size for quantization analysis.

Usage:
    pip install matplotlib
    python examples/plot_kld_analysis.py

Edit the DATA section below to add your models.
"""

import argparse

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    raise SystemExit(1)


# --- Edit your data here ---
# Format: (file_size_gib, mean_kld, label)
DATA = [
    (30.0, 0.0, "Original (Llama-3.1-8B bf16)"),
    (6.2, 0.033707, "Quantized (FP8_INT4)"),
    (5.4, 0.076226, "Quantized (W4A16_GS128)"),
    (5.7, 0.109275, "Quantized (NVFP4)"),
    (8.6, 0.000899, "Quantized (W8A16_GS128)"),
    (8.5, 0.006547, "Quantized (W8A8-FP8_BLOCK)"),
]
# Add more: DATA.append((size_gib, kld, "Label"))


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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Unique style per point: original=gray circle, quants=distinct markers+colors
    QUANT_MARKERS = ["s", "^", "D", "v", "p", "h", "8", "*", "P", "X"]
    QUANT_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for i, (s, k, lbl) in enumerate(DATA):
        if k == 0:
            color, marker = "gray", "o"
        else:
            idx = (i - 1) % len(QUANT_MARKERS)
            color = QUANT_COLORS[idx % len(QUANT_COLORS)]
            marker = QUANT_MARKERS[idx % len(QUANT_MARKERS)]
        legend_label = f"{lbl} ({k:.4f})" if k > 0 else lbl
        ax.scatter(s, k, label=legend_label, color=color, s=120, marker=marker, zorder=3)

    ax.set_xlabel("File Size (GiB)")
    ax.set_ylabel("Mean KL Divergence (Lower is Better)")
    ax.set_title("Quantization Quality: KLD vs File Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

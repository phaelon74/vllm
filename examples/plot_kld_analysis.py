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

    sizes = [d[0] for d in DATA]
    klds = [d[1] for d in DATA]
    labels = [d[2] for d in DATA]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["gray" if k == 0 else "blue" for k in klds]
    markers = ["o" if k == 0 else "s" for k in klds]

    for i, (s, k, lbl) in enumerate(DATA):
        ax.scatter(s, k, label=lbl, color=colors[i], s=120, marker=markers[i], zorder=3)
        ax.annotate(
            f"{k:.4f}",
            (s, k),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

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

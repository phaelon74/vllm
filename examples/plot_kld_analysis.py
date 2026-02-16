#!/usr/bin/env python3
"""
Plot KLD (Kullback-Leibler Divergence) vs model file size for quantization analysis.

Categories: INT AWQ (W4A16, W8A16, FP8-INT4), NVFP4, GGUF - each with distinct color/shape.

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


# --- Edit your data here ---
# Format: (file_size_gib, mean_kld, bpw, label, category)
# category: "original" | "int_awq" | "nvfp4" | "gguf"
DATA = [
    # Original
    (30.0, 0.0, 16.0, "Original (Llama-3.1-8B bf16)", "original"),
    # INT AWQ: W4A16, W8A16, FP8-INT4
    (5.4, 0.076226, 4.25, "W4A16_GS128", "int_awq"),
    (8.6, 0.000899, 8.25, "W8A16_GS128", "int_awq"),
    (6.2, 0.033707, 4.0, "FP8_INT4", "int_awq"),
    # NVFP4
    (5.7, 0.109275, 4.0, "NVFP4", "nvfp4"),
    (8.5, 0.006547, 8.0, "W8A8-FP8_BLOCK", "nvfp4"),
    # GGUF (from reference chart)
    (3.5, 0.1241, 3.50, "IQ3_XS", "gguf"),
    (3.65, 0.1782, 3.64, "Q3_K_S", "gguf"),
    (4.0, 0.0744, 4.00, "Q3_K_M", "gguf"),
    (4.4, 0.0327, 4.42, "IQ4_XS", "gguf"),
    (4.7, 0.0305, 4.67, "Q4_K_S", "gguf"),
    (4.9, 0.0267, 4.89, "Q4_K_M", "gguf"),
    (5.6, 0.0102, 5.57, "Q5_K_S", "gguf"),
    (5.7, 0.0092, 5.70, "Q5_K_M", "gguf"),
    (6.6, 0.0040, 6.56, "Q6_K", "gguf"),
    (8.0, 0.0011, 8.50, "Q8_0", "gguf"),
]

# Category styles: (color, marker)
STYLES = {
    "original": ("#7f7f7f", "o"),
    "int_awq": ("#1f77b4", "s"),
    "nvfp4": ("#ff7f0e", "^"),
    "gguf": ("#2ca02c", "D"),
}


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

    fig, ax = plt.subplots(figsize=(16, 10))

    # Track legend entries to avoid duplicates
    legend_added = set()

    # Stagger label positions to reduce overlap (alternate quadrants)
    offsets = [
        (25, 20), (25, -30), (-100, 20), (25, 20), (-100, -30),
        (25, -30), (-100, 20), (25, 20), (-100, -30), (25, 20),
        (-100, -30), (25, -30), (-100, 20), (25, 20), (-100, -30),
        (25, -30),
    ]

    for i, (size_gib, kld, bpw, label, category) in enumerate(DATA):
        color, marker = STYLES[category]
        full_label = f"Llama-3.1-8B-Instruct-{label}" if category != "original" else label

        # Add to legend only once per category
        if category not in legend_added:
            legend_label = {
                "original": "Original (bf16)",
                "int_awq": "INT AWQ (W4A16, W8A16, FP8-INT4)",
                "nvfp4": "NVFP4 / W8A8-FP8",
                "gguf": "GGUF",
            }[category]
            ax.scatter(
                [], [], color=color, marker=marker, s=120, label=legend_label
            )
            legend_added.add(category)

        ax.scatter(size_gib, kld, color=color, s=120, marker=marker, zorder=3)

        # Annotate with model name, KLD, bpw (like reference graph)
        text = f"{full_label}\n{kld:.4f}\n{bpw:.2f}bpw"

        if category == "original":
            xytext = (-120, 15)
        else:
            xytext = offsets[i % len(offsets)]
        ha = "left" if xytext[0] > 0 else "right"
        va = "bottom" if xytext[1] > 0 else "top"
        ax.annotate(
            text,
            (size_gib, kld),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("File Size (GiB)", fontsize=12)
    ax.set_ylabel("Mean KL Divergence (Lower is Better)", fontsize=12)
    ax.set_title(
        "Llama-3.1-8B-Instruct Quantization Analysis: Mean KL Divergence vs. Model File Size",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=-0.02)
    ax.set_xlim(left=2.5, right=32)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Illustrate truncation and exact folding using an analytic one-pole response.

Run with ``uv run python scripts/make_fig_response_folding.py [--output DIR]``. This schematic
uses h[n] = r**n, r = exp(-1/32), and N = 16; it contains no campaign data.
The folded response is the geometric sum r**n / (1 - r**N), anchored to
H(omega) = 1 / (1 - r exp(-i omega)) at omega = 2 pi k / N.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from noise_stitching_data.paths import STYLE, parse

BLOCK_LENGTH = 16
POLE = np.exp(-1 / 32)


def main() -> None:
    """Export the vector schematic at the paper's figure width."""
    _, output = parse(__doc__.splitlines()[0])
    (output / "figures").mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(15 / 2.54, 8.5 / 2.54), layout="constrained")
        n = np.arange(4 * BLOCK_LENGTH + 1)
        left.plot(n / BLOCK_LENGTH, POLE**n, color="#0072B2", label=r"Full response $h[n]$")
        left.axvspan(1, 4, color="0.93")
        left.axvline(1, color="0.4", linestyle=":")
        left.text(0.5, 0.05, "Kept", ha="center")
        left.text(2.5, 0.8, "Tail discarded\nby truncation", ha="center")
        left.set(xlabel=r"Sample index $n/N$", ylabel="Impulse response", xlim=(0, 4), ylim=(0, 1.12))
        left.set_xticks([0, 1, 2, 3, 4])
        left.legend(loc="upper right", frameon=False)
        left.text(0, 1.04, "(a)", transform=left.transAxes)

        m = np.arange(BLOCK_LENGTH)
        folded = POLE**m / (1 - POLE**BLOCK_LENGTH)
        right.plot(m / BLOCK_LENGTH, folded, color="#0072B2", marker="o", markersize=2.5, label="Sum of all blocks")
        right.plot(m / BLOCK_LENGTH, POLE**m, color="#D55E00", linestyle="--", label="First block only")
        for j, style in [(1, ":"), (2, "-.")]:
            right.plot(
                m / BLOCK_LENGTH,
                POLE ** (m + j * BLOCK_LENGTH),
                color="0.4",
                linestyle=style,
                label=rf"Block {j + 1}, shifted left",
            )
        right.set(xlabel=r"Position in first block $n/N$", ylabel="Impulse response", xlim=(0, 1), ylim=(0, 2.8))
        right.set_xticks([0, 0.5, 1])
        right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        right.text(0, 1.04, "(b)", transform=right.transAxes)
        fig.savefig(
            output / "figures" / "fig_response_folding.pdf",
            metadata={"Creator": "scripts/make_fig_response_folding.py", "CreationDate": None},
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

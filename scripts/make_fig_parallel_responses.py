#!/usr/bin/env python3
"""Draw the manuscript's parallel-responses figure from the window form.

The figure illustrates the "parallel responses give one" regime of the section
on the coherence as a cosine between two window responses. At ``O = P`` the
cyclic spectrum is ``S_k(f) = sum_q c_q a_q conj(b_q) / (P f_s)`` with
``a_q = G(f + alpha_k / 2 - nu_q)`` and ``b_q = G(f - alpha_k / 2 - nu_q)``,
where ``G`` is the response of the window ``g`` (the rising blend weight
followed by the falling one) and ``c_q`` are the target's line weights on the
window's frequency grid. The coherence is the cosine between ``a`` and ``b``
in the inner product weighted by ``c``.

The target is a flat band, lines ``W/8 <= |q| <= 3W/8`` at equal weight, and
every frequency is measured in window bins ``Delta f = f_s / W`` from the
lowest in-band line. ``G`` is evaluated by direct summation, with no
approximation.

Panel (a): the target and the stitched output's spectrum ``S_0`` for the
gwmock-noise sequence at ``P = 64``, with the range of panel (b) shaded and the
frequencies of panels (c) and (d) marked.

Panel (b): ``rho_1`` across the lower band edge, at whole-bin offsets, for both
phase sequences at ``P = 64`` and for the gwmock-noise sequence at ``P = 256``.

Panels (c) and (d): the two responses on each in-band line, scaled by the
line weight and normalized to unit norm, at 8 bins outside the edge and at the
first in-band line. Outside the edge the two profiles coincide (the responses
are parallel); inside it they peak on different lines.

Usage::

    uv run python scripts/make_fig_parallel_responses.py [--output DIR]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from noise_stitching_data.blend import BlendWeights, gwmock_blend_weights, lal_blend_weights
from noise_stitching_data.paths import STYLE, parse

FIGURE_WIDTH = 15.0 / 2.54
FIGURE_HEIGHT = 10.0 / 2.54

PDF_METADATA = {
    "Title": "Coherence near a band-mask edge and the two window responses",
    "Creator": "scripts/make_fig_parallel_responses.py",
    "Producer": "matplotlib",
    "CreationDate": None,
}

#: Cyclic frequency index plotted throughout.
K = 1

#: Offsets from the lowest in-band line, in window bins, for panel (a). They
#: are whole bins: half-way between bins the sine window's response vanishes on
#: every in-band line outside the edge, and the cosine there is 0/0.
OFFSETS = np.arange(-12, 5)

#: Offsets of panels (b) and (c), and the in-band lines they show.
OUTSIDE_OFFSET = -8
INSIDE_OFFSET = 0
LINES_SHOWN = 24

#: Frequencies of panel (a), in window bins from zero frequency.
SPECTRUM_BINS = np.arange(0.0, 64.0 + 1e-9, 1.0 / 8.0)


def window(weights: BlendWeights) -> np.ndarray:
    """Return the window ``g``: the rising blend weight, then the falling one.

    Args:
        weights: Blend weights with ``window_size == 2 * stride``.

    Returns:
        ``g`` of length ``window_size``.
    """
    return np.concatenate([weights.weight_new, weights.weight_old])


def band_lines(window_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the flat band's line indices, line weights and lowest in-band line.

    Args:
        window_size: Window length ``W`` in samples.

    Returns:
        ``(q, c, q_edge)``: every line index on the window grid, its weight,
        and the index of the lowest in-band positive-frequency line.
    """
    q = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    q_edge = window_size // 8
    c = np.where((np.abs(q) >= q_edge) & (np.abs(q) <= 3 * window_size // 8), 1.0, 0.0)
    return q, c, q_edge


def responses(g: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two window responses on every line, and the line weights.

    Args:
        g: The window.
        offset: Frequency ``f`` in window bins from the lowest in-band line.

    Returns:
        ``(a, b, c)`` on the lines of :func:`band_lines`.
    """
    window_size = g.size
    stride = window_size // 2
    q, c, q_edge = band_lines(window_size)
    f = (q_edge + offset) / window_size
    half_alpha = 0.5 * K / stride
    r = np.arange(window_size)

    def response(nu: np.ndarray) -> np.ndarray:
        return np.exp(-2j * np.pi * np.outer(nu, r)) @ g

    return response(f + half_alpha - q / window_size), response(f - half_alpha - q / window_size), c


def coherence(g: np.ndarray, offset: float) -> float:
    """Return ``rho_K`` at ``offset`` window bins from the band edge.

    Args:
        g: The window.
        offset: Frequency in window bins from the lowest in-band line.

    Returns:
        The cosine between the two responses in the line-weighted inner product.
    """
    a, b, c = responses(g, offset)
    inner = np.sum(c * a * np.conj(b))
    return float(np.abs(inner) / np.sqrt(np.sum(c * np.abs(a) ** 2) * np.sum(c * np.abs(b) ** 2)))


def profiles(g: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the normalized response profiles on the first in-band lines.

    Args:
        g: The window.
        offset: Frequency in window bins from the lowest in-band line.

    Returns:
        ``(j, a_hat, b_hat, ratio)``: line index from the edge, ``sqrt(c) |a|``
        and ``sqrt(c) |b|`` normalized to unit norm over all lines, and
        ``a / b`` on the lines shown.
    """
    a, b, c = responses(g, offset)
    q, _, q_edge = band_lines(g.size)
    a_hat = np.sqrt(c) * np.abs(a) / np.sqrt(np.sum(c * np.abs(a) ** 2))
    b_hat = np.sqrt(c) * np.abs(b) / np.sqrt(np.sum(c * np.abs(b) ** 2))
    shown = (q >= q_edge) & (q < q_edge + LINES_SHOWN)
    return q[shown] - q_edge, a_hat[shown], b_hat[shown], a[shown] / b[shown]


def output_spectrum(g: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Return the stitched output's spectrum ``S_0``, relative to the band center.

    Args:
        g: The window.
        bins: Frequencies in window bins from zero frequency.

    Returns:
        ``sum_q c_q |G(f - nu_q)|^2`` at each frequency, divided by its value at
        the band center.
    """
    window_size = g.size
    q, c, q_edge = band_lines(window_size)
    r = np.arange(window_size)

    def power(nu: float) -> float:
        response = np.exp(-2j * np.pi * np.outer(nu - q / window_size, r)) @ g
        return float(np.sum(c * np.abs(response) ** 2))

    center = power(2 * q_edge / window_size)
    return np.array([power(b / window_size) for b in bins]) / center


def draw() -> plt.Figure:
    """Return the figure, drawn with the current style.

    Returns:
        The four-panel figure.
    """
    g64 = window(gwmock_blend_weights(128, 64))
    curves = (
        (g64, r"\texttt{gwmock-noise}, $P = 64$", "tab:blue", "-", "o"),
        (window(lal_blend_weights(128, 64)), r"\texttt{LALSimulation}, $P = 64$", "tab:orange", (0, (4, 2)), "s"),
        (window(gwmock_blend_weights(512, 256)), r"\texttt{gwmock-noise}, $P = 256$", "black", (0, (1, 1.5)), "x"),
    )
    figure, panels = plt.subplot_mosaic(
        [["spectrum"] * 3, ["coherence", "outside", "edge"]],
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        layout="constrained",
        width_ratios=(1.5, 1, 1),
        height_ratios=(1, 1.1),
    )
    _, _, q_edge = band_lines(g64.size)
    top = panels["spectrum"]
    top.axvspan(q_edge, 3 * q_edge, color="0.9", linewidth=0)
    range_left = q_edge + OFFSETS[0] - 0.5
    range_right = q_edge + OFFSETS[-1] + 0.5
    top.annotate(
        "",
        xy=(range_left, 3e-7),
        xytext=(range_right, 3e-7),
        arrowprops={"arrowstyle": "|-|", "color": "0.35", "linewidth": 0.7},
    )
    top.text((range_left + range_right) / 2, 5e-7, "Range of (b)", ha="center", color="0.35")
    top.text(32, 0.03, "In band", ha="center", color="0.35")
    target_edges = np.array([0.0, q_edge, q_edge, 3 * q_edge, 3 * q_edge, 64.0])
    top.plot(target_edges, [1e-12, 1e-12, 1.0, 1.0, 1e-12, 1e-12], color="0.5", linestyle=(0, (4, 2)), label="Target")
    spectrum = output_spectrum(g64, SPECTRUM_BINS)
    top.plot(SPECTRUM_BINS, spectrum, color="tab:blue", label=r"Output $S_0$, \texttt{gwmock-noise}, $P = 64$")
    for offset, label, shift, factor in ((OUTSIDE_OFFSET, "(c)", -3.0, 30.0), (INSIDE_OFFSET, "(d)", 5.0, 0.03)):
        level = output_spectrum(g64, np.array([q_edge + offset]))[0]
        top.annotate(
            label,
            xy=(q_edge + offset, level),
            xytext=(q_edge + offset + shift, level * factor),
            ha="center",
            arrowprops={"arrowstyle": "->", "linewidth": 0.6},
        )
    top.set_yscale("log")
    top.set_ylim(1e-7, 1e1)
    top.set_xlim(0.0, 64.0)
    top.set_xlabel(r"$f/\Delta f$")
    top.set_ylabel("Spectrum")
    top.set_title("(a)", loc="left")
    top.legend(loc="upper right")

    left = panels["coherence"]
    for g, label, color, style, marker in curves:
        rho = [coherence(g, x) for x in OFFSETS]
        left.plot(OFFSETS, rho, color=color, linestyle=style, marker=marker, markersize=3.0, mfc="none", label=label)
    left.axvspan(0.0, OFFSETS[-1] + 0.5, color="0.9", linewidth=0)
    left.set_yscale("log")
    left.set_ylim(1e-4, 1.5)
    left.set_xlim(OFFSETS[0] - 0.5, OFFSETS[-1] + 0.5)
    left.set_xlabel(r"$(f - f_{\mathrm{edge}})/\Delta f$")
    left.set_ylabel(r"$\rho_1$")
    left.set_title("(b)", loc="left")
    left.text(2.0, 0.3, "In band", ha="center", color="0.35")
    left.legend(loc="lower left")

    for axis, offset, title in (
        (panels["outside"], OUTSIDE_OFFSET, "(c)"),
        (panels["edge"], INSIDE_OFFSET, "(d)"),
    ):
        j, a_hat, b_hat, _ = profiles(g64, offset)
        axis.plot(j - 0.15, a_hat, "o", markersize=2.5, color="tab:blue", label=r"$\sqrt{c_q}\,|a_q|$")
        axis.plot(j + 0.15, b_hat, "s", markersize=2.5, color="tab:red", mfc="none", label=r"$\sqrt{c_q}\,|b_q|$")
        axis.vlines(j - 0.15, 0.0, a_hat, color="tab:blue", linewidth=0.6)
        axis.vlines(j + 0.15, 0.0, b_hat, color="tab:red", linewidth=0.6)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"In-band line $j$")
        axis.set_title(title, loc="left")
        axis.text(0.95, 0.6, f"$\\rho_1 = {coherence(g64, offset):.2f}$", transform=axis.transAxes, ha="right")
    panels["outside"].set_ylabel("Normalized response")
    panels["outside"].legend(loc="upper right")
    return figure


def report() -> None:
    """Print every number the manuscript quotes from this figure."""
    for name, g in (
        ("gwmock P=64", window(gwmock_blend_weights(128, 64))),
        ("lal P=64", window(lal_blend_weights(128, 64))),
        ("gwmock P=256", window(gwmock_blend_weights(512, 256))),
    ):
        values = ", ".join(f"{x:+d}: {coherence(g, x):.3g}" for x in (-8, -4, -1, 0, 1, 2))
        print(f"{name}: rho_1 at offsets {values}")
    g64 = window(gwmock_blend_weights(128, 64))
    for offset in (OUTSIDE_OFFSET, INSIDE_OFFSET):
        _, a_hat, b_hat, ratio = profiles(g64, offset)
        print(
            f"offset {offset:+d}: top-line share of |a|^2 {np.max(a_hat**2):.3f}, of |b|^2 {np.max(b_hat**2):.3f}; "
            f"a/b at j=0,3,10: {np.round(ratio[[0, 3, 10]], 3)}; arg(a/b) spread {np.ptp(np.angle(ratio)):.2e} rad"
        )


def main() -> int:
    """Draw the figure and print its numbers.

    Args:
    Returns:
        Zero.
    """
    _, output_dir = parse(__doc__.splitlines()[0])
    output = output_dir / "figures" / "fig_parallel_responses.pdf"
    # With LaTeX text, matplotlib reads the font family when it renders, so
    # the style must hold for the save as well as for the drawing.
    with plt.style.context(STYLE):
        figure = draw()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, metadata=PDF_METADATA)
    plt.close(figure)
    report()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

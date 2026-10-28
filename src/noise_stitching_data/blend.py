"""Blend weights and the phase modulation they impose on the chunk covariance.

The appendix shows that the covariance of the stitched process factorises: the
covariance of two output samples is the chunk autocovariance at their lag, times
a purely geometric factor built from the blend weights at the two samples' phases
within the stitching period. This module holds the weights and that factor.

The factor, not the chunk autocovariance, is where the construction's whole
second-moment signature lives. It is one at zero lag for any square-complementary
weight pair --- which is why the stitched process has exactly the right variance
everywhere and no variance test can see the defect --- and it is identically zero
beyond twice the stride, which truncates the process's covariance however
long-ranged the target's is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Largest departure from unity tolerated in ``w_old**2 + w_new**2``.
SQUARE_COMPLEMENTARY_TOLERANCE = 1e-12

#: Shortest overlap a phase ramp can be defined on: one sample gives no interval
#: to advance the phase across.
MINIMUM_RAMP_SAMPLES = 2


@dataclass(frozen=True)
class BlendWeights:
    """One crossfade's geometry: window length, stride, and the two weight arrays.

    Window ``n`` covers output samples ``[n * stride, n * stride + window_size)``.
    Its first ``overlap_size = window_size - stride`` samples are blended with the
    tail of window ``n - 1``; the rest are copied through. Writing a sample's
    dependence on the two windows it can draw on as ``x = beta * old + alpha *
    new`` collapses the blended and copied cases into one expression, and
    :attr:`alpha` and :attr:`beta` are that expression's coefficients as a
    function of the sample's phase within the stride.

    Attributes:
        window_size: Synthesis window length in samples.
        stride: Spacing between consecutive window starts in samples. It is the
            period of the resulting cyclostationary process.
        weight_old: Weight applied to the older window across the overlap, of
            length ``window_size - stride``.
        weight_new: Weight applied to the newer window across the overlap, of the
            same length.
    """

    window_size: int
    stride: int
    weight_old: np.ndarray
    weight_new: np.ndarray

    def __post_init__(self) -> None:
        """Check the geometry and the square-complementary condition.

        Raises:
            ValueError: If the stride does not leave a positive overlap no longer
                than itself, if the weight arrays do not have the overlap's
                length, or if the weights are not square-complementary.
        """
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive integers.")
        overlap_size = self.window_size - self.stride
        if overlap_size <= 0:
            raise ValueError("stride must be shorter than window_size, so that consecutive windows overlap.")
        if overlap_size > self.stride:
            raise ValueError(
                "the overlap must not exceed the stride: this package covers the regime in which at most two "
                f"windows contribute to any one sample (window_size={self.window_size}, stride={self.stride})."
            )
        for name, weights in (("weight_old", self.weight_old), ("weight_new", self.weight_new)):
            if weights.shape != (overlap_size,):
                raise ValueError(f"{name} must have shape ({overlap_size},), got {weights.shape}.")
        departure = np.max(np.abs(self.weight_old**2 + self.weight_new**2 - 1.0))
        if departure > SQUARE_COMPLEMENTARY_TOLERANCE:
            raise ValueError(
                f"the weights are not square-complementary: max |w_old^2 + w_new^2 - 1| = {departure:.3e}."
            )

    @property
    def overlap_size(self) -> int:
        """Overlap length in samples.

        Returns:
            ``window_size - stride``.
        """
        return self.window_size - self.stride

    @property
    def alpha(self) -> np.ndarray:
        """Coefficient of the newer window, by phase within the stride.

        Returns:
            An array of length ``stride``: the blend weight across the overlap,
            and one across the copied interior beyond it.
        """
        coefficients = np.ones(self.stride, dtype=float)
        coefficients[: self.overlap_size] = self.weight_new
        return coefficients

    @property
    def beta(self) -> np.ndarray:
        """Coefficient of the older window, by phase within the stride.

        Returns:
            An array of length ``stride``: the blend weight across the overlap,
            and zero across the copied interior beyond it.
        """
        coefficients = np.zeros(self.stride, dtype=float)
        coefficients[: self.overlap_size] = self.weight_old
        return coefficients


def gwmock_blend_weights(window_size: int, stride: int | None = None) -> BlendWeights:
    """Return the blend weights the gwmock-noise coloured simulator applies.

    Its blend phase runs ``theta_j = (pi / 2) * j / (overlap_size - 1)``, so the
    fade reaches exactly zero and exactly one at the ends of the overlap. The
    generator then divides the blended samples by ``sqrt(cos^2 + sin^2)``, which
    is identically one in exact arithmetic; the division is folded into the
    weights here so that the reference stitcher reproduces the generator's
    arithmetic to round-off rather than to within the size of that no-op.

    Args:
        window_size: Synthesis window length in samples.
        stride: Spacing between window starts. Defaults to
            ``window_size - window_size // 2``, which is what the generator uses.

    Returns:
        The blend weights.

    Raises:
        ValueError: If the resulting overlap is shorter than two samples, where
            the phase ramp is not defined.
    """
    if stride is None:
        stride = window_size - window_size // 2
    overlap_size = window_size - stride
    if overlap_size < MINIMUM_RAMP_SAMPLES:
        raise ValueError(f"the gwmock-noise phase ramp needs an overlap of at least two samples, got {overlap_size}.")
    phase = np.linspace(0.0, np.pi / 2.0, overlap_size)
    weight_old, weight_new = np.cos(phase), np.sin(phase)
    norm = np.sqrt(weight_old**2 + weight_new**2)
    return BlendWeights(window_size, stride, weight_old / norm, weight_new / norm)


def lal_blend_weights(window_size: int, stride: int | None = None) -> BlendWeights:
    """Return the blend weights LALSimulation's noise generator applies.

    Its blend phase runs ``theta_j = pi * j / (2 * overlap_size)``, one sample
    short of the ramp above: the fade never quite reaches zero or one inside the
    blended region. The two are square-complementary at every sample all the
    same, and differ only by a rescaling of the phase slope by
    ``overlap_size / (overlap_size - 1)``, but they are different constructions
    and the manuscript specialises the derivation to each.

    Args:
        window_size: Synthesis window length in samples.
        stride: Spacing between window starts. Defaults to
            ``window_size - window_size // 2``.

    Returns:
        The blend weights.
    """
    if stride is None:
        stride = window_size - window_size // 2
    overlap_size = window_size - stride
    phase = np.pi * np.arange(overlap_size) / (2.0 * overlap_size)
    return BlendWeights(window_size, stride, np.cos(phase), np.sin(phase))


def covariance_support_half_width(weights: BlendWeights) -> int:
    """Return the lag beyond which the stitched covariance is identically zero.

    Args:
        weights: The crossfade geometry.

    Returns:
        ``2 * stride``. Two output samples further apart than this cannot share a
        synthesis window, and the windows are independent, so their covariance is
        exactly zero whatever the target autocovariance does at that lag.
    """
    return 2 * weights.stride


def modulation_factor(weights: BlendWeights, phase: int, lags: np.ndarray) -> np.ndarray:
    """Return the geometric factor multiplying the chunk autocovariance.

    For an output sample at phase ``phase`` within the stride and a second sample
    ``lag`` samples later, the covariance of the pair is this factor times the
    chunk autocovariance at ``lag``. The three non-zero branches are the three
    ways the pair can share a window: both drawing on the same pair of windows,
    or the later sample's older window being the earlier sample's newer one, or
    the reverse.

    Args:
        weights: The crossfade geometry.
        phase: Phase of the earlier-indexed sample within the stride, in
            ``[0, stride)``.
        lags: Integer lags, of any shape.

    Returns:
        An array of the same shape as ``lags``.

    Raises:
        ValueError: If ``phase`` is outside ``[0, stride)``.
    """
    if not 0 <= phase < weights.stride:
        raise ValueError(f"phase must lie in [0, {weights.stride}), got {phase}.")
    lags = np.asarray(lags, dtype=np.int64)
    alpha, beta = weights.alpha, weights.beta

    offset = phase + lags
    block = np.floor_divide(offset, weights.stride)
    other_phase = offset - block * weights.stride

    factor = np.zeros(lags.shape, dtype=float)
    same = block == 0
    later = block == 1
    earlier = block == -1
    factor[same] = alpha[other_phase[same]] * alpha[phase] + beta[other_phase[same]] * beta[phase]
    factor[later] = beta[other_phase[later]] * alpha[phase]
    factor[earlier] = alpha[other_phase[earlier]] * beta[phase]
    return factor


def stitched_autocovariance(
    weights: BlendWeights,
    chunk_autocovariance: np.ndarray,
    phase: int,
    lags: np.ndarray,
) -> np.ndarray:
    """Return the closed-form covariance of the stitched process.

    Args:
        weights: The crossfade geometry.
        chunk_autocovariance: The circulant chunk autocovariance, of length
            ``weights.window_size``, as returned by
            :func:`noise_stitching_data.chunk_covariance.circulant_autocovariance`.
        phase: Phase of the earlier-indexed sample within the stride.
        lags: Integer lags, of any shape.

    Returns:
        An array of the same shape as ``lags``, giving
        ``Cov(x[t + lag], x[t])`` for any ``t`` congruent to ``phase`` modulo the
        stride.

    Raises:
        ValueError: If ``chunk_autocovariance`` does not have the window's length.
    """
    chunk_autocovariance = np.asarray(chunk_autocovariance, dtype=float)
    if chunk_autocovariance.shape != (weights.window_size,):
        raise ValueError(
            f"chunk_autocovariance must have shape ({weights.window_size},), got {chunk_autocovariance.shape}."
        )
    lags = np.asarray(lags, dtype=np.int64)
    return modulation_factor(weights, phase, lags) * chunk_autocovariance[np.abs(lags) % weights.window_size]

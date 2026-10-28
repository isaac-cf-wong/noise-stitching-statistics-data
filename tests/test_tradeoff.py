"""The trade-off join and the Pareto front."""

from __future__ import annotations

from noise_stitching_data.tradeoff import TradeoffPoint, join, label_of, pareto_front, unmatched, window_seconds


def _point(name: str, error: float, speed: float) -> TradeoffPoint:
    return TradeoffPoint(name, "x", error, speed, None)


def test_the_front_keeps_only_the_points_nothing_beats() -> None:
    """A slower and less accurate point is dropped; ties on one axis keep the better of the two."""
    points = [
        _point("a", 0.1, 1e6),
        _point("b", 0.01, 1e5),
        _point("c", 0.2, 5e5),
        _point("d", 0.1, 2e6),
        _point("e", 0.01, 1e5),
    ]
    names = [point.name for point in pareto_front(points)]
    assert names == ["b", "e", "d"]


def test_equal_points_both_survive_and_the_empty_set_has_an_empty_front() -> None:
    """Two identical points do not dominate each other."""
    assert len(pareto_front([_point("a", 0.1, 1.0), _point("b", 0.1, 1.0)])) == 2
    assert pareto_front([]) == []


def test_join_matches_on_name_and_configuration_and_reports_the_rest() -> None:
    """Rows join on their configuration; a failed fit or a missing cost row is reported, not invented."""
    exact = [
        {"name": "ar_levinson", "parameters": {"order": 256}, "eps_total": 0.3},
        {"name": "ar_levinson", "parameters": {"order": 4096}, "failure": "FitError: x"},
        {"name": "overlap_save_fir", "parameters": {"filter_length": 99}, "eps_total": 0.5},
        {"name": "crossfade_gwmock", "parameters": {"grid": "64s_50"}, "eps_total": 1e-3},
        {"name": "crossfade_gwmock", "parameters": {"grid": "8s_50"}, "eps_total": 5e-2},
        {"name": "crossfade_lal", "parameters": {"grid": "64s_50"}, "eps_total": 1e-3},
    ]
    costs = [
        {
            "name": "ar_levinson",
            "parameters": {"order": 256},
            "status": "ran",
            "samples_per_second": 6e6,
            "state_bytes": 2480,
        },
        {
            "name": "crossfade_gwmock",
            "parameters": {"implementation": "gwmock"},
            "status": "ran",
            "samples_per_second": 1.3e7,
            "state_bytes": 271,
        },
    ]
    points = join(exact, costs, crossfade_grids=["64s_50"])
    assert [(p.name, p.label) for p in points] == [("ar_levinson", "p=256"), ("crossfade_gwmock", "64 s")]
    assert unmatched(exact, points) == ["overlap_save_fir L=99", "crossfade_gwmock 8 s"]


def test_labels_and_window_parsing() -> None:
    """Labels read as the configuration; the window parses from the grid id."""
    assert label_of("arma_state_space", {"ar_order": 256, "ma_order": 64}) == "256/64"
    assert label_of("pycbc_reproducible", {"filter_duration": 128.0}) == "128 s"
    assert window_seconds("2048s_50_8192") == 2048.0


def test_windows_are_told_apart_by_overlap_and_take_their_own_throughput() -> None:
    """A 25 % overlap is not labelled as a 50 % one, and a window measured on its own is not given another's speed."""
    exact = [
        {"name": "crossfade_gwmock", "parameters": {"grid": "64s_50"}, "eps_total": 1e-3},
        {"name": "crossfade_gwmock", "parameters": {"grid": "64s_25"}, "eps_total": 2e-3},
        {"name": "crossfade_gwmock", "parameters": {"grid": "256s_50"}, "eps_total": 1e-4},
    ]
    costs = [
        {"name": "crossfade_gwmock", "parameters": {}, "status": "ran", "samples_per_second": 1e7, "state_bytes": 271}
    ]
    points = join(exact, costs, crossfade_throughput={"256s_50": 4.6e6})
    assert [p.label for p in points] == ["64 s", "64 s, 25%", "256 s"]
    assert [p.samples_per_second for p in points] == [1e7, 1e7, 4.6e6]
    assert len({(p.name, p.label) for p in points}) == 3

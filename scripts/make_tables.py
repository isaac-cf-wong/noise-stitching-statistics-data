#!/usr/bin/env python3
r"""Write the paper's generated numbers and tables.

Reads the archived results under ``<data>/results`` and the spectra under ``<data>/psd``, and
writes ``<output>/generated/numbers.tex`` (the ``\rv`` macro and every number the text
quotes) and one ``<output>/generated/tab_*.tex`` per table.

Usage::

    uv run python scripts/make_tables.py [--data DIR] [--output DIR]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from noise_stitching_data.dependence import crossfade_curve, edge_slope, floor_of, window_for_error
from noise_stitching_data.manuscript_inputs import (
    MAIN_ARMS,
    TABLE_WINDOWS,
    errors_by_row,
    key_of,
    load,
    math_number,
    number_definitions,
    row_key,
    significant,
    tabular,
)
from noise_stitching_data.paths import parse
from noise_stitching_data.spectra import analysis_band, load_psd

#: The rows of the error tables, in order: ``(method, configuration, label)``.
METHOD_ROWS: tuple[tuple[str, str, str], ...] = (
    *(("crossfade_gwmock", f"{w:g}", f"Crossfade, ${w:g}$~s window") for w in TABLE_WINDOWS),
    *(("overlap_save_fir", str(n), f"FIR, ${n}$ taps") for n in (256, 1024, 4096, 16384, 65536)),
    *(("ar_levinson", str(p), f"AR, order ${p}$") for p in (64, 256, 1024, 4096)),
    *(("arma_state_space", pair, f"ARMA, ${pair}$") for pair in ("64/16", "256/64", "1024/256")),
    *(("pycbc_reproducible", str(d), rf"\texttt{{PyCBC}} filter, ${d}$~s") for d in (16, 64, 128)),
)

#: The sampling rate of the window-requirement table.
WINDOW_RATE = 4096

#: The window of the table comparing the edge conventions, in seconds.
COMPARISON_WINDOW = 1024.0

#: A scale error and the deviation, in sampling spreads, the implementation check is asked to detect it at.
SCALE_ERROR = 0.1
POWER_THRESHOLD = 8.0

#: Windows from which the crossfade's throughput is quoted as one range, in seconds.
LONG_WINDOW = 256.0

#: The segment index the stored state and the rebuild cost are quoted at, and the size of a kilobyte.
SEGMENT = 100
KILOBYTE = 1000

#: The detector arms shown in the window-requirement table, with the label each is given.
WINDOW_TABLE_ARMS: tuple[tuple[str, str], ...] = (
    ("aligo_o4high_smooth", "aLIGO O4 projected"),
    ("aligo_o3h1_lines", "aLIGO O3 H1, with lines"),
    ("aligo_aplus", "LIGO A+ (and LIGO-India)"),
    ("virgo_o4", "Advanced Virgo O4"),
    ("virgo_design", "Advanced Virgo design"),
    ("kagra_o4", "KAGRA 25~Mpc"),
    ("kagra_design", "KAGRA design"),
    ("ce_40", "Cosmic Explorer"),
    ("ce_wideband", "Cosmic Explorer wideband"),
    ("et_d_smooth", "ET-D"),
    ("et10_full_cryo_lines", "ET-10 cryogenic, with lines"),
)
THRESHOLDS = (1e-2, 1e-3, 1e-4)


def error_table(results: Path, edge: str, rate: int) -> str:
    """Return the table of total covariance errors for the main arms at one edge convention."""
    snapshots = [errors_by_row(load(results / "exact" / f"{arm}__{edge}__{rate}.json")) for arm, _ in MAIN_ARMS]
    rows = []
    for method, configuration, label in METHOD_ROWS:
        cells = []
        for snapshot in snapshots:
            row = snapshot.get((method, configuration))
            cells.append(
                "---" if row is None else ("fit failed" if "failure" in row else math_number(row["eps_total"]))
            )
        rows.append([label, *cells])
    return tabular(
        ["l", *["r"] * len(MAIN_ARMS)], ["Method", *[name for _, name in MAIN_ARMS]], rows, rules_after=(5, 10, 14, 17)
    )


def window_table(results: Path, spectra: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return the window-requirement table over the detector curves, and the numbers its prose quotes."""
    rows, numbers = [], []
    for arm, label in WINDOW_TABLE_ARMS:
        snapshot = load(results / "exact" / f"{arm}__specified__4096.json")
        curve = crossfade_curve(snapshot["results"]["exact_errors"])
        low = analysis_band(arm)[0]
        table_frequencies, table_values = load_psd(spectra, arm)
        slope = edge_slope(np.asarray(table_frequencies), np.asarray(table_values), low)
        windows = [window_for_error(curve, threshold) for threshold in THRESHOLDS]
        cells = [label, f"${low:g}$", f"${slope:.1f}$"]
        cells += [math_number(curve[w]) if w in curve else "---" for w in (16.0, 64.0, 256.0, 1024.0)]
        cells += [f"$>{max(curve):g}$" if w is None else f"${w:.0f}$" for w in windows]
        rows.append(cells)
        for threshold, window in zip(THRESHOLDS, windows, strict=True):
            numbers.append((key_of("window", arm, f"{threshold:g}"), "---" if window is None else f"{window:.0f}"))
        numbers.append((key_of("floor", arm), math_number(floor_of(curve))))
        numbers.append((key_of("slope", arm), f"{slope:.1f}"))
    header = [
        "Curve",
        r"$f_{\rm low}$ (Hz)",
        "$s$",
        "$16$~s",
        "$64$~s",
        "$256$~s",
        "$1024$~s",
        "$T_{10^{-2}}$",
        "$T_{10^{-3}}$",
        "$T_{10^{-4}}$",
    ]
    return tabular(["l", "r", "r", *["r"] * 4, *["r"] * 3], header, rows), numbers


def error_numbers(results: Path, edge: str, rate: int) -> list[tuple[str, str]]:
    """Return one macro per error of the main arms, for the prose to quote."""
    entries = []
    harmonics = 0
    arms = WINDOW_TABLE_ARMS if edge == "specified" and rate == WINDOW_RATE else MAIN_ARMS
    for arm, _ in arms:
        for (method, configuration), row in errors_by_row(
            load(results / "exact" / f"{arm}__{edge}__{rate}.json")
        ).items():
            if "failure" in row:
                continue
            base = (arm, edge, rate, method, configuration)
            entries.append((key_of("eps", *base), math_number(row["eps_total"])))
            entries.append((key_of("epsstat", *base), math_number(row["eps_stationary"])))
            entries.append((key_of("epscyc", *base), math_number(row["eps_cyclic"])))
            harmonics = max(harmonics, int(row.get("diagnostic", {}).get("harmonics", 0)))
            if "eps_total_to_nyquist" in row:
                entries.append((key_of("epsnyq", *base), math_number(row["eps_total_to_nyquist"])))
    entries.append((key_of("harmonics", edge, rate), str(harmonics)))
    return entries


def cost_entries(results: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return the cost and state table of the aLIGO smooth arm and the numbers the prose quotes."""
    costs = {
        row_key(r): r
        for r in load(results / "costs" / "aligo_o4high_smooth__64s_50.json")["results"]["costs"]
        if r["status"] == "ran"
    }
    stored: dict[tuple[str, str], tuple[int, float]] = {}
    for r in load(results / "reproducibility.json")["results"]["reproducibility"]:
        if r["arm"] == "aligo_o4high_smooth" and r["segment_index"] == SEGMENT:
            stored[row_key(r)] = (r["stored_state_bytes"], r["no_cache_seconds"])
    entries, rows = [], []
    for method, configuration, label in METHOD_ROWS:
        if method == "arma_state_space" or (method == "crossfade_gwmock" and configuration != "64"):
            continue
        cost = costs.get((method, "" if method == "crossfade_gwmock" else configuration))
        state, rebuild = stored.get((method, "" if method == "crossfade_gwmock" else configuration), (None, None))
        if method == "crossfade_gwmock":
            state, rebuild = stored.get((method, ""), (None, None))
        base = (method, configuration)
        cells = [label]
        if cost is None:
            cells.append("---")
        else:
            speed = cost["samples_per_second"] / 1e6
            cells.append(math_number(speed))
            entries.append((key_of("speed", *base), significant(speed)))
        if state is None:
            cells += ["---", "---"]
        else:
            cells += [state_text(state), math_number(rebuild)]
            entries += [(key_of("state", *base), state_text(state)), (key_of("rebuild", *base), significant(rebuild))]
        held = None if cost is None else cost.get("state_bytes")
        cells.append("---" if held is None else state_text(held))
        if held is not None:
            entries.append((key_of("held", *base), state_text(held)))
        rows.append(cells)
    header = ["Method", "Throughput (M samples/s)", "Stored snapshot", "Rebuild at $n=100$ (s)", "Held while running"]
    return tabular(["l", "r", "r", "r", "r"], header, rows, rules_after=(0, 5, 9)), entries


def state_text(nbytes: int) -> str:
    """Return a byte count for a table, in bytes below one kilobyte and in kilobytes above."""
    return rf"\ensuremath{{{nbytes}}}~B" if nbytes < KILOBYTE else rf"\ensuremath{{{nbytes / KILOBYTE:.1f}}}~kB"


def implementation_entries(results: Path) -> list[tuple[str, str]]:
    """Return the worst standardized residual and the comparison counts of the implementation checks."""
    worst, count, cells = 0.0, 0, 0
    by_method: dict[str, float] = {}
    for path in sorted((results / "implementation").glob("*.json")):
        cells += 1
        for check in load(path)["results"]["implementation_checks"]:
            for band in check["bands"]:
                count += 1
                z = abs(band["z_score"])
                worst = max(worst, z)
                by_method[check["name"]] = max(by_method.get(check["name"], 0.0), z)
    spreads = [
        abs((band["ratio"] - 1.0) / band["z_score"])
        for path in sorted((results / "implementation").glob("*.json"))
        for check in load(path)["results"]["implementation_checks"]
        for band in check["bands"]
        if band["z_score"] != 0.0
    ]
    # What a scale error of ten per cent would read as, in the sampling spreads these runs actually had.
    ten_percent = sorted(SCALE_ERROR / spread for spread in spreads)
    entries = [("impl-worst", f"{worst:.1f}"), ("impl-comparisons", str(count)), ("impl-cells", str(cells))]
    entries += [
        ("impl-power-median", f"{ten_percent[len(ten_percent) // 2]:.0f}"),
        ("impl-power-fraction", f"{100.0 * sum(z > POWER_THRESHOLD for z in ten_percent) / len(ten_percent):.0f}"),
    ]
    entries += [(key_of("impl-worst", name), f"{z:.1f}") for name, z in by_method.items()]
    return entries


def commit_entries(results: Path) -> list[tuple[str, str]]:
    """Return the commits of the campaign: the analysis code and the generator, read from a snapshot."""
    snapshot = load(next(iter(sorted((results / "exact").glob("*.json")))))
    return [
        ("analysis-commit", rf"\commit{{{snapshot['analysis_commit']}}}"),
        ("generator-commit", rf"\commit{{{snapshot['generator_commit']}}}"),
    ]


def derived_entries(results: Path) -> list[tuple[str, str]]:
    """Return the ratios and ranges the prose quotes, derived from the snapshots."""
    entries = []
    for arm, _ in MAIN_ARMS:
        rows = errors_by_row(load(results / "exact" / f"{arm}__specified__4096.json"))
        for (method, configuration), row in rows.items():
            if method == "crossfade_gwmock" and "-" not in configuration and row["eps_cyclic"] > 0.0:
                entries.append(
                    (key_of("ratio", arm, configuration), significant(row["eps_stationary"] / row["eps_cyclic"]))
                )
        fir, crossfade = rows[("overlap_save_fir", "65536")], rows[("crossfade_gwmock", "64")]
        entries.append((key_of("ratio-fir", arm), significant(fir["eps_total"] / crossfade["eps_total"])))
    windows = {}
    for arm, _ in WINDOW_TABLE_ARMS:
        curve = crossfade_curve(load(results / "exact" / f"{arm}__specified__4096.json")["results"]["exact_errors"])
        for threshold in THRESHOLDS:
            window = window_for_error(curve, threshold)
            if window is not None:
                windows.setdefault(threshold, []).append(window)
    for threshold, values in windows.items():
        entries += [
            (key_of("window-min", f"{threshold:g}"), f"{min(values):.0f}"),
            (key_of("window-max", f"{threshold:g}"), f"{max(values):.0f}"),
        ]
    speeds = []
    for path in sorted((results / "costs").glob("aligo_o4high_smooth__*s_50__crossfade_gwmock.json")):
        window = float(path.name.split("__")[1].split("s_")[0])
        record = next(r for r in load(path)["results"]["costs"] if r["name"] == "crossfade_gwmock")
        entries.append((key_of("speed-win", f"{window:g}"), significant(record["samples_per_second"] / 1e6)))
        entries.append((key_of("rss-win", f"{window:g}"), f"{record['peak_rss_bytes'] / 1e9:.2f}"))
        if window >= LONG_WINDOW:
            speeds.append(record["samples_per_second"] / 1e6)
    if speeds:
        entries.append(("speed-windows", f"{min(speeds):.1f} to {max(speeds):.1f}"))
    reproducibility = load(results / "reproducibility.json")["results"]["reproducibility"]
    entries.append(("repro-records", str(sum("bit_exact" in r for r in reproducibility))))
    spans = [r["joint_request_max_relative_difference"] for r in reproducibility if "same_request_bit_exact" in r]
    entries += [("pycbc-diff-min", math_number(min(spans))), ("pycbc-diff-max", math_number(max(spans)))]
    snapshot = load(next(iter(sorted((results / "implementation").glob("*.json")))))
    entries.append(("impl-seconds", f"{snapshot['config']['segments'] * snapshot['config']['segment_seconds']:.0f}"))
    return entries


FINE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("aligo_aplus", "aligo_aplus_fine", "LIGO A+ (and LIGO-India)"),
    ("virgo_design", "virgo_design_fine", "Advanced Virgo design"),
    ("kagra_design", "kagra_design_fine", "KAGRA design"),
)


def fine_table(results: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return the table of window requirements of the three design curves at two tabulation spacings."""
    rows, entries = [], []
    for coarse, fine, label in FINE_PAIRS:
        for arm, spacing in ((coarse, "1/8"), (fine, "1/32")):
            curve = crossfade_curve(load(results / "exact" / f"{arm}__specified__4096.json")["results"]["exact_errors"])
            windows = [window_for_error(curve, threshold) for threshold in THRESHOLDS]
            cells = [label if spacing == "1/8" else "", rf"${spacing}$"]
            cells += [f"$>{max(curve):g}$" if w is None else f"${w:.0f}$" for w in windows]
            rows.append(cells)
            for threshold, window in zip(THRESHOLDS, windows, strict=True):
                entries.append(
                    (key_of("window-spacing", arm, f"{threshold:g}"), "---" if window is None else f"{window:.0f}")
                )
    header = ["Curve", r"Spacing (Hz)", r"$T_{10^{-2}}$", r"$T_{10^{-3}}$", r"$T_{10^{-4}}$"]
    return tabular(["l", "r", "r", "r", "r"], header, rows, rules_after=(1, 3)), entries


def edge_comparison_table(results: Path) -> str:
    """Return the table comparing the specified and the hard edge, and the Nyquist guard, over all curves."""
    rows = []
    for arm, label in WINDOW_TABLE_ARMS:
        specified = errors_by_row(load(results / "exact" / f"{arm}__specified__4096.json"))
        hard = errors_by_row(load(results / "exact" / f"{arm}__hard__4096.json"))
        wide = errors_by_row(load(results / "exact" / f"{arm}__specified__8192.json"))
        crossfade = specified[("crossfade_gwmock", f"{COMPARISON_WINDOW:g}")]
        rows.append(
            [
                label,
                math_number(crossfade["eps_total"]),
                math_number(hard[("crossfade_gwmock", f"{COMPARISON_WINDOW:g}")]["eps_total"]),
                math_number(specified[("overlap_save_fir", "65536")]["eps_total"]),
                math_number(hard[("overlap_save_fir", "65536")]["eps_total"]),
                math_number(crossfade["eps_total_to_nyquist"]),
                math_number(wide[("crossfade_gwmock", "2048")]["eps_total"]),
            ]
        )
    header = [
        "Curve",
        r"Crossfade, specified",
        r"Crossfade, hard",
        r"FIR $65536$, specified",
        r"FIR $65536$, hard",
        r"Crossfade to Nyquist",
        r"Crossfade, $8192$~Hz",
    ]
    return tabular(["l", *["r"] * 6], header, rows)


def main() -> None:
    """Write the generated files."""
    data, output = parse(__doc__.splitlines()[0])
    results = data / "results"
    output = output / "generated"
    output.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str]] = []
    for edge in ("specified", "hard"):
        (output / f"tab_errors_{edge}.tex").write_text(error_table(results, edge, 4096), encoding="utf-8")
        entries += error_numbers(results, edge, 4096)
    entries += error_numbers(results, "specified", 8192)
    table, numbers = window_table(results, data)
    (output / "tab_windows.tex").write_text(table, encoding="utf-8")
    entries += numbers
    cost_table, cost_numbers = cost_entries(results)
    (output / "tab_costs.tex").write_text(cost_table, encoding="utf-8")
    entries += cost_numbers + implementation_entries(results) + commit_entries(results) + derived_entries(results)
    (output / "tab_edges.tex").write_text(edge_comparison_table(results), encoding="utf-8")
    fine_text, fine_numbers = fine_table(results)
    (output / "tab_fine.tex").write_text(fine_text, encoding="utf-8")
    entries += fine_numbers
    (output / "numbers.tex").write_text(number_definitions(entries), encoding="utf-8")
    print(f"wrote {len(entries)} numbers and 6 tables to {output}")


if __name__ == "__main__":
    main()

# Data release: choosing a generator for streaming Gaussian noise

This repository reproduces the tables and figures of the paper _Choosing a
generator for streaming Gaussian noise in gravitational-wave mock data:
accuracy, cost and reproducibility_ (Isaac C. F. Wong) from its archived
results.

The results are on Zenodo:
[10.5281/zenodo.23243286](https://doi.org/10.5281/zenodo.23243286). The record
holds the measured results behind the paper's tables, data figures and the
numbers quoted from them (covariance errors, throughput and state,
implementation checks, reproducibility, the closed-form verification and the
band-edge coherence), with their configuration, the seeds of the sampled results or the rule that
derives them, the commits that produced them, and the noise spectra the results
were computed for. Its `README.md` describes the layout, the common fields and the measured
quantities of each kind of result.

The noise itself was generated with
[gwmock-noise](https://github.com/Leuven-Gravity-Institute/gwmock-noise), at
the commit recorded in each result file.

## Reproducing the paper

You need [uv](https://docs.astral.sh/uv/) and a LaTeX installation, since the
figures set their text with LaTeX. Then

```console
make
```

which runs, in order:

```console
uv run python scripts/fetch_data.py                 # download, check and unpack the record into data/
uv run python scripts/make_tables.py                # output/generated/
uv run python scripts/make_fig_response_folding.py  # output/figures/
uv run python scripts/make_fig_parallel_responses.py
uv run python scripts/make_fig_windows.py
uv run python scripts/make_fig_tradeoff.py
uv run python scripts/make_fig_c1_residuals.py
```

Each script takes `--data DIR` (default `data/`) and `--output DIR` (default
`output/`). `fetch_data.py` checks every downloaded file against the MD5
checksum Zenodo publishes for it. `--archive results.tar.gz` unpacks a copy you
already have instead, without a checksum to check it against.

| output                                     | in the paper                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `generated/numbers.tex`                    | the numbers the text quotes through `\rv{<key>}` macros                  |
| `generated/tab_errors_specified.tex`       | covariance errors of every generator, smoothly continued band edge        |
| `generated/tab_errors_hard.tex`            | the same with the spectrum cut to zero below the band                     |
| `generated/tab_costs.tex`                  | throughput, stored state and rebuild cost                                 |
| `generated/tab_windows.tex`                | the crossfade window each detector curve needs                            |
| `generated/tab_fine.tex`                   | the same for three design curves at two tabulation spacings               |
| `generated/tab_edges.tex`                  | the band-edge conventions and the Nyquist guard compared                  |
| `figures/fig_response_folding.pdf`         | truncating an impulse response against folding it                        |
| `figures/fig_parallel_responses.pdf`       | the coherence as the cosine between two window responses                  |
| `figures/fig_windows.pdf`                  | the crossfade's error against window length for every detector curve     |
| `figures/fig_tradeoff_specified.pdf`       | accuracy against throughput for every generator                           |
| `figures/fig_tradeoff_hard.pdf`            | the same with the hard band edge (not shown in the paper)                 |
| `generated/tradeoff-points.json`           | the points of the two trade-off figures                                   |
| `figures/fig_c1_residuals.pdf`             | the measured crossfade covariance against its closed form                 |

The two schematic figures, `fig_response_folding` and `fig_parallel_responses`,
are computed analytically and read no data. The numbers of the closed-form
verification are printed by `make_fig_c1_residuals.py`, and those of the
band-edge study can be read from `results/edge-leakage/`.

The six generated tables and `numbers.tex` reproduce the paper's byte for byte. The figures
render identically to the paper's with the pinned library versions; the PDF
files themselves may differ in their internal byte layout.

## Layout

- `scripts/`: the fetch script and one script per output.
- `src/noise_stitching_data/`: what the scripts share: reading the result
  files, the crossfade's error against window length, the accuracy–throughput
  join, the crossfade blend weights, the analysis bands and spectra, and the
  LaTeX table writers.
- `tests/`: unit tests of that code, and checks that the tables match the
  published ones and that the band table matches the data. These two run only
  once the data has been fetched, so CI, which does not fetch it, skips them.

## Citing

Please cite the paper, and the data release as

> I. C. F. Wong, _Results: Choosing a generator for streaming Gaussian noise
> in gravitational-wave mock data_, Zenodo (2026),
> doi:[10.5281/zenodo.23243286](https://doi.org/10.5281/zenodo.23243286).

## Licence

The code is under the BSD 3-Clause licence (`LICENSE`). The data on Zenodo is
under CC BY 4.0.

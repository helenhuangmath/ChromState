# `tools/` — run other segmentation tools and score them all the same way

`chromstate compare` benchmarks ChromState against *reimplementations* of other methods'
statistical models. That is fast, runs anywhere, and isolates the modelling choice — but
it is not evidence about the real software, and `docs/PLAN.md` §4b commits to re-running
the ChromHMM row against real ChromHMM before publication.

This folder is where that happens. It runs the actual tools on ChromState's bins, maps
their output back onto those bins, and scores everything with one metric registry.

```
tools/
  run_tools.py     entry point (no install needed — puts src/ and tools/ on the path)
  scaling.py       entry point for runtime/memory against dataset size
  tools.json       external tools: command template + output parser, disabled until configured
  setup/           recreate the conda env, patches for current R/Armadillo, env wrapper
  toolbench/
    methods/       EVERY COMPARED METHOD LIVES HERE
      internal.py    ChromState, ChromHMM-like, Segway-like, GMM, k-means, naive signal
      chromhmm.py    real ChromHMM (java -jar ChromHMM.jar LearnModel)          [verified]
      epicseg.py     real EpiCSeg (normalizecounts + segment)                   [verified]
      normr.py       real normR (enrichR per mark; patterns + bivalency only)   [verified]
      dihmm.py       real diHMM (ykai16 C++ build; nucleosome + domain level)   [verified]
      configured.py  anything described by a command template in tools.json
      __init__.py    assembles the registry from the modules above
    r/             R helpers the adapters call (normR runner, EpiCSeg posterior export)
    drivers/       non-R helpers (diHMM train + annotate)
    core.py        ToolRun / ToolAdapter / registry / subprocess timing and memory
    io_formats.py  export bins to each tool's format; read segmentations back by coordinate
    scores.py      post-hoc bivalency / activity scores for tools that only return states
    metrics.py     the metric registry — truth-based and truth-free
    evaluate.py    run each tool, score each run
    scaling.py     runtime and memory against dataset size
    report.py      tables, figures, REPORT.md
    cli.py         check / metrics / run
```

Results of running all of this are in [`docs/tool_comparison.md`](../docs/tool_comparison.md).

Everything in the comparison is defined under `toolbench/methods/`, whichever kind it is,
and importing that package registers all of it — so `check` and the runner see one flat
namespace of method keys. The *statistical implementations* of the six in-process methods
stay in `src/chromstate/benchmark/baselines.py`, because `chromstate compare` ships in the
wheel and cannot import from this repo-local folder; `methods/internal.py` holds the
adapter, not the model.

**Adding a method.** If it is a command line, add an entry to `tools.json` and write no
Python. If it needs real logic — a multi-step tool, a bespoke output format — add a module
to `methods/` defining an adapter plus a `build()` that returns instances, and list it in
`METHOD_MODULES`.

## Use

```bash
python tools/run_tools.py check                  # what can run here, and how to enable the rest
python tools/run_tools.py metrics                # what each metric measures and what it needs
python tools/run_tools.py run -o compare_out     # simulated data with known truth
python tools/run_tools.py run -o out --counts counts.tsv          # real data, truth-free metrics
python tools/run_tools.py run --tools chromstate,chromhmm --dry-run   # print commands, run nothing
```

`run` writes `comparison.csv`, `agreement.csv`, `REPORT.md`, and figures in the same
style as the rest of the package. Useful flags: `--metrics ari,cost` to select metrics,
`--n-states`, `--seed`, `--timeout` (per tool), `--reference` (which tool the agreement
metric compares against), `--per-bin` to dump every tool's call per bin for IGV.

## The three kinds of tool

| Kind | What it is | Trust |
|---|---|---|
| **in-process** | ChromState and the reimplemented baselines from `chromstate.benchmark` | exact and reproducible, but evidence about *models*, not software |
| **real tools** | ChromHMM, EpiCSeg, normR, diHMM — dedicated adapters in `methods/` | evidence about the tool; each was executed and its output mapping checked here |
| **configured** | anything in `tools.json`: command template + output parser | evidence about the tool, once you have checked the command |

`verified` means the adapter has been executed against the real software in this
repository and its output mapping checked against the tool's own files (for example,
ChromHMM's segment BED against its posterior argmax). ChromHMM 1.28, EpiCSeg 1.0, normR
1.36.0 and diHMM (ykai16 C++) are verified; what running them changed is in each
adapter's docstring. Entries in `tools.json` (Segway, epigenome-ssm) ship with
`verified: false`: their command lines are transcribed from documentation and have
**not** been run. `check` prints the flag, and `REPORT.md` warns about any unverified tool
whose numbers it prints.

## Enabling a tool

The quickest route is the isolated environment below: `tools/setup/setup_env.sh`
installs all four tools, and `tools/setup/benchrun.sh <command>` runs any command with
every one of them switched on. Individually:

| Tool | Environment variables | Notes |
|---|---|---|
| ChromHMM | `CHROMHMM_JAR` (+ `JAVA_HOME` or `java` on PATH); optional `CHROMHMM_ASSEMBLY`, `CHROMHMM_PROCESSORS`, `CHROMHMM_BINARY_DIR` | runs `LearnModel ... -noenrich -printposterior` |
| EpiCSeg | `EPICSEG_R` = launcher from `epicseg:::getLauncher()`; optional `EPICSEG_NTHREADS` | launcher is run with the Rscript in its own shebang |
| normR | `NORMR_RSCRIPT` = an Rscript that can load `normr`; optional `NORMR_PROCS` | seeded Poisson pseudo-input when there is no control |
| diHMM | `DIHMM_BUILD` = dir with `dihmm_ext.so`; `DIHMM_PYTHON`; optional `DIHMM_DOMAIN_SIZE` | bin and domain state counts are both `--n-states` |

```bash
export CHROMHMM_JAR=/path/to/ChromHMM.jar      # java via JAVA_HOME or PATH
python tools/run_tools.py run --tools chromstate,chromhmm -o out
```

The adapter feeds ChromHMM the binarised input ChromState writes, so both rows see
identical bins and identical presence calls and the only difference is the model. To
benchmark ChromHMM's *own* binarisation too, run its `BinarizeBed` yourself and point
`CHROMHMM_BINARY_DIR` at the result.

**Anything else** — edit `tools.json`. Each entry is a command template plus a description
of where the output lands:

```json
{
  "key": "mytool",
  "display_name": "My segmenter (what it is, spelled out)",
  "enabled": true,
  "requires_env": ["MYTOOL_BIN"],
  "requires_binary": "mytool",
  "input_format": "count_matrix",
  "command": ["{env:MYTOOL_BIN}", "--in", "{counts}", "--states", "{n_states}", "--out", "{outdir}"],
  "output": {"format": "segmentation_bed", "path": "{outdir}/segments.bed", "per_sample": true}
}
```

Placeholders: `{workdir} {outdir} {counts} {regions} {binary_dir} {n_states} {seed}
{bin_size} {sample} {env:VAR}`. Input formats: `count_matrix`, `regions_bed`,
`chromhmm_binary`. Output formats: `segmentation_bed` (BED4, mapped back by coordinate)
or `labels` (one label per exported bin, in row order); add `posterior_path` if the tool
writes a per-bin posterior matrix and you want the calibration metrics.

Shipped-but-disabled entries for **Segway** and **epigenome-ssm** are starting points —
the closest prior art named in `docs/PLAN.md` §0 that has not been run here. Segway needs
GMTK, which is Linux-only; on macOS it needs a linux/amd64 container (see below).

## Recreating the benchmark environment

Everything external lives in one conda env, `chromstate-bench`, plus downloads under
`~/tools/chromstate-bench/`; the base env, the system R library and the user's R library
are never touched. This is what ran on macOS arm64 (10 cores, 16 GB):

```bash
bash tools/setup/setup_env.sh                       # env + downloads + R builds + diHMM build
bash tools/setup/benchrun.sh python tools/run_tools.py check   # all four should say available
```

`setup_env.sh` does, in order:

1. `mamba create -p ~/miniconda3/envs/chromstate-bench -c conda-forge -c bioconda` with
   Python 3.11, OpenJDK 17, R 4.5, the Bioconductor dependencies that have osx-arm64
   builds (Rsamtools, Rhtslib, GenomicRanges, rtracklayer, qvalue, edgeR, …), compilers,
   Armadillo, OpenBLAS (OpenMP variant, for `cblas.h`), Boost.Python, numba and psutil.
2. Downloads pinned versions: ChromHMM 1.28 (`ChromHMM.zip` release), EpiCSeg @383b069,
   kfoots @4fb7c59, diHMM (ykai16) @c5a4bd9.
3. Patches for current toolchains, each a few characters and none changing a model:
   `epicseg_fcone.patch` (BLAS `FCONE` arguments, R ≥ 4.3), `kfoots_r43.patch`
   (`&&` on a vector is an error since R 4.3), and three diHMM patches applied by
   `build_dihmm.sh` — an off-by-one emission loop, a row/column `each_col` product, and
   a Viterbi backtrace that shifted every label by one bin.
4. `install_r_tools.R`: `bamsignals` and `normr` from Bioconductor source (no osx-arm64
   conda build), the patched kfoots and EpiCSeg, and the EpiCSeg launcher.
5. `build_dihmm.sh`: one `clang++` command instead of upstream's Python-2.7 CMake setup,
   serial (upstream's OpenMP loop has a data race), Armadillo checks compiled out.

**Segway** was not installed: GMTK has no macOS build, and the Docker daemon on the
benchmark machine was not running. With a working Docker, a `linux/amd64` image with
`conda install -c bioconda segway` is the route; the `tools.json` entry then needs a
genomedata archive built from the exported counts and a second step, `segway annotate`
(called `identify` before Segway 3).

Reproduce the published comparison and scaling runs:

```bash
bash tools/setup/benchrun.sh python tools/run_tools.py run -o compare_sim \
    --simulate 25000 --n-chroms 4 --blur 0.3 --seed 0 --timeout 1800
bash tools/setup/benchrun.sh python tools/scaling.py -o scaling \
    --sizes 6250,25000,100000 --tools chromstate,chromhmm,epicseg,normr,dihmm
```

## Metrics

`python tools/run_tools.py metrics` prints the registry. Two families:

- **Truth-based** (simulated data, or an orthogonal ground truth): `recovery` (adjusted
  Rand, adjusted mutual information, V-measure, matched accuracy), `bivalency` (area
  under the precision-recall curve), `magnitude` (does the score track the *amount* of
  mark), `calibration` (expected calibration error, overconfidence, saturated calls).
- **Truth-free** (works on real data): `segmentation` (segment count and length, states
  used, occupancy entropy), `coverage`, `confidence` (how decisive the posterior is,
  regardless of correctness), `agreement` (adjusted Rand against a reference tool),
  `cost` (runtime; peak memory when `psutil` is installed).

Truth-free metrics say a tool is *consistent*, *decisive*, or *cheap* — never that it is
right. Two tools agreeing is equally consistent with both being wrong.

Add a metric by adding one entry to `METRICS` in `toolbench/metrics.py`; it declares what
it requires (`truth`, `posterior`, `reference`, …) and is skipped automatically when a run
cannot supply it.

## Rules the harness keeps

1. **The bin grid never changes.** Every tool sees the same intervals. Output is mapped
   back by genomic coordinate, never by row order — tools merge bins into segments and
   reorder chromosomes, and a row-order assumption silently shifts everything by one bin.
2. **Bins a tool did not call stay unassigned**, and coverage is reported as a metric
   instead of hiding in a denominator.
3. **Nothing is dropped.** A tool that is missing, times out, or crashes gets a row with
   a status and a reason. A table that silently omits the failures reads as though
   everything ran.
4. **Tools run one at a time**, because runtime and memory are measurements of the tools
   and tools sharing a machine measure each other.
5. **`—` in a report means not measurable**, not zero.

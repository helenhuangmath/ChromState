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
  tools.json       external tools: command template + output parser, disabled until configured
  toolbench/
    methods/       EVERY COMPARED METHOD LIVES HERE
      internal.py    ChromState, ChromHMM-, Segway-, EpiCSeg-, EpiSegMix-, diHMM-, normR-like, GMM, k-means, naive
      chromhmm.py    real ChromHMM (java -jar ChromHMM.jar LearnModel)
      configured.py  anything described by a command template in tools.json
      __init__.py    assembles the registry from the modules above
    core.py        ToolRun / ToolAdapter / registry / subprocess timing and memory
    io_formats.py  export bins to each tool's format; read segmentations back by coordinate
    metrics.py     the metric registry — truth-based and truth-free
    evaluate.py    run each tool, score each run
    report.py      tables, figures, REPORT.md
    cli.py         check / metrics / run
```

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
| **real ChromHMM** | `java -jar ChromHMM.jar LearnModel` on the same bins | evidence about the tool; enable with `CHROMHMM_JAR` |
| **configured** | anything in `tools.json`: command template + output parser | evidence about the tool, once you have checked the command |

The real-ChromHMM adapter is **verified**: it has been run against ChromHMM 1.28
(`jernst98/ChromHMM`) on simulated bins, which is how the `-noenrich` flag was found to be
necessary (`LearnModel` writes the segmentation and posteriors, then aborts in enrichment
when the bins do not match the named assembly). It also reads `emissions_K.txt` to give
real ChromHMM a bivalency and activity score with the same formula as every other method.

Every other external adapter ships with `verified: false`, and `check` prints it. It means the
command line was transcribed from the tool's published documentation and has **not** been
executed against the real software in this repository. Run `--dry-run` first, read the
command, fix `tools.json` if the installed version disagrees, and flip `verified` to true
once you have seen it work. `REPORT.md` carries the same warning for any unverified tool
whose numbers it prints.

## Enabling a tool

**ChromHMM** — install it, then:

```bash
export CHROMHMM_JAR=/path/to/ChromHMM.jar      # java must be on PATH
export CHROMHMM_ASSEMBLY=hg38                  # optional, default hg19
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

Shipped-but-disabled entries for **EpiCSeg**, **Segway**, and **epigenome-ssm** are
starting points — the closest prior art named in `docs/PLAN.md` §0, and the rows a
reviewer will ask for.

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

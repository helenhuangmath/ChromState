"""Tests for the external-tool comparison harness in ``tools/``.

The external adapters cannot be exercised against the real software here, so the tests
target what actually breaks silently: coordinate alignment when a tool returns segments
instead of bins, and the promise that a tool which fails still appears in the table. A
stub "tool" -- a shell script that writes a BED -- stands in for the real thing and
exercises the whole export/run/parse/score path.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import numpy as np
import pytest

import chromstate as cs
from chromstate.simulate import simulate

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from toolbench import evaluate_runs, registry, run_tools  # noqa: E402
from toolbench.core import UNASSIGNED, ToolRun  # noqa: E402
from toolbench.evaluate import agreement_matrix, availability_table  # noqa: E402
from toolbench.io_formats import (  # noqa: E402
    export_count_matrix,
    labels_from_segmentation_bed,
)
from toolbench.methods import ExternalCommandAdapter  # noqa: E402
from toolbench.metrics import METRICS, TRUTH_FREE  # noqa: E402


@pytest.fixture(scope="module")
def small():
    ds, truth = simulate(n_bins=250, n_chroms=2, seed=11, blur=0.2)
    cs.normalize(ds, method="median_ratio")
    return ds, truth


# -- registry ----------------------------------------------------------------


def test_every_internal_tool_is_available_and_verified():
    table = availability_table()
    internal = table.loc[table["kind"] == "in-process"]
    assert len(internal) >= 5
    assert internal["available"].all()
    assert internal["verified"].all(), "bundled models run in-process; nothing to verify"


def test_every_compared_method_comes_from_the_methods_folder():
    """One folder defines the comparison; nothing may register from anywhere else."""
    from toolbench import methods

    built = {a.key for a in methods.build_all()}
    assert built == set(registry()), (
        "a method is registered that toolbench/methods/ does not build"
    )
    for module in methods.METHOD_MODULES:
        assert callable(getattr(module, "build", None)), module.__name__
        assert module.build(), f"{module.__name__}.build() returned nothing"


def test_external_tools_are_registered_and_flagged_honestly():
    table = availability_table().set_index("tool")
    # These adapters were executed against the real software (see docs/tool_comparison.md).
    for key in ("chromhmm", "epicseg", "normr", "dihmm"):
        assert key in table.index
        assert bool(table.loc[key, "verified"]), key
    # Command templates in tools.json have not been, and must say so.
    for key in ("segway", "epigenome_ssm"):
        assert not bool(table.loc[key, "verified"]), key
    # An unavailable tool must still explain itself.
    for key in table.index[~table["available"].astype(bool)]:
        assert table.loc[key, "detail"]
        assert table.loc[key, "how_to_enable"]


# -- coordinate alignment ----------------------------------------------------


def test_segmentation_bed_maps_onto_bins_by_coordinate(small, tmp_path):
    """Segments, not bins, and out of order: the alignment must still be exact."""
    ds, _ = small
    sample = ds.samples[0]
    bins = ds.bins.loc[ds.sample_mask(sample)].reset_index(drop=True)
    chrom = bins["chrom"].iloc[0]
    sub = bins.loc[bins["chrom"] == chrom]
    size = int(sub["end"].iloc[0] - sub["start"].iloc[0])

    # Two segments spanning bins 10-19 and 40-44, written in reverse coordinate order.
    lo, hi = int(sub["start"].iloc[10]), int(sub["start"].iloc[20])
    lo2, hi2 = int(sub["start"].iloc[40]), int(sub["start"].iloc[45])
    bed = tmp_path / "seg.bed"
    bed.write_text(
        f"{chrom}\t{lo2}\t{hi2}\tE2\n"
        f"{chrom}\t{lo}\t{hi}\tE1\n"
    )

    states, vocab = labels_from_segmentation_bed(ds, str(bed), sample=sample)
    rows = np.flatnonzero(ds.sample_mask(sample))
    chrom_rows = rows[(bins["chrom"] == chrom).to_numpy()]

    assert vocab == ["E2", "E1"]
    assert (states[chrom_rows[10:20]] == vocab.index("E1")).all()
    assert (states[chrom_rows[40:45]] == vocab.index("E2")).all()
    # Everything else is unassigned, not silently zero.
    assert states[chrom_rows[20:40]].tolist() == [UNASSIGNED] * 20
    assert (states == UNASSIGNED).sum() == ds.n_rows - 15
    assert size > 0


def test_a_tool_that_covers_nothing_reports_zero_coverage():
    run = ToolRun(tool="x", display_name="X", states=np.full(10, UNASSIGNED))
    assert run.coverage == 0.0


# -- running and scoring -----------------------------------------------------


def test_internal_tools_run_and_score_against_truth(small):
    ds, truth = small
    runs = run_tools(ds, ["chromstate", "kmeans"], n_states=truth.n_states, seed=0,
                     workroot=None, verbose=False)
    assert all(r.ok for r in runs)
    table = evaluate_runs(runs, ds, truth=truth, reference="chromstate")

    assert list(table["tool"]) == ["chromstate", "kmeans"]
    for column in ("ari", "matched_accuracy", "bin_coverage", "runtime_s"):
        assert table[column].notna().all(), column
    # ChromState is scored against itself, so it gets no agreement value.
    assert np.isnan(table.loc[table["tool"] == "chromstate", "agreement_ari"].iloc[0])
    assert 0.0 <= table.loc[table["tool"] == "kmeans", "agreement_ari"].iloc[0] <= 1.0


def test_truth_free_metrics_work_without_truth(small):
    ds, _ = small
    runs = run_tools(ds, ["kmeans"], n_states=5, seed=0, verbose=False)
    table = evaluate_runs(runs, ds, truth=None, metrics=TRUTH_FREE)
    assert "ari" not in table.columns, "recovery needs truth and must not appear"
    assert table["mean_segment_kb"].notna().all()
    assert table["bin_coverage"].iloc[0] == pytest.approx(1.0)


def test_unavailable_tool_becomes_a_row_not_a_missing_row(small):
    ds, truth = small
    runs = run_tools(ds, ["kmeans", "chromhmm"], n_states=truth.n_states, verbose=False)
    table = evaluate_runs(runs, ds, truth=truth)
    assert len(table) == 2
    row = table.loc[table["tool"] == "chromhmm"].iloc[0]
    assert row["status"] == "skipped"
    assert row["reason"]
    assert np.isnan(row["ari"])


def test_agreement_matrix_is_symmetric_with_unit_diagonal(small):
    ds, truth = small
    runs = run_tools(ds, ["chromstate", "kmeans", "gmm"], n_states=truth.n_states,
                     verbose=False)
    mat = agreement_matrix(runs)
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat.values), 1.0)
    np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-12)


def test_every_metric_declares_its_requirements():
    for name, metric in METRICS.items():
        assert metric.description.strip(), name
        assert metric.columns, name
        assert set(metric.requires) <= {"truth", "posterior", "reference", "bivalency",
                                        "activity"}, name


# -- the external path, via a stub tool --------------------------------------


def _write_stub_tool(tmp_path) -> str:
    """A fake segmenter: reads the exported counts, writes a two-state BED."""
    script = tmp_path / "stub_tool.py"
    script.write_text(
        "import sys, csv\n"
        "counts, outdir = sys.argv[1], sys.argv[2]\n"
        "rows = list(csv.DictReader(open(counts), delimiter='\\t'))\n"
        "import os; os.makedirs(outdir, exist_ok=True)\n"
        "with open(os.path.join(outdir, 'segmentation.bed'), 'w') as fh:\n"
        "    for r in rows:\n"
        "        mark = float(r[[k for k in r if k not in ('chrom','start','end','sample')][0]])\n"
        "        fh.write(f\"{r['chrom']}\\t{r['start']}\\t{r['end']}\\t\"\n"
        "                 f\"{'HIGH' if mark > 0 else 'LOW'}\\n\")\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_configured_external_tool_runs_end_to_end(small, tmp_path):
    ds, truth = small
    stub = _write_stub_tool(tmp_path)
    spec = {
        "key": "stub",
        "display_name": "Stub segmenter (test double)",
        "enabled": True,
        "input_format": "count_matrix",
        "command": [sys.executable, stub, "{counts}", "{outdir}"],
        "output": {"format": "segmentation_bed", "path": "{outdir}/segmentation.bed",
                   "per_sample": True},
        "verified": False,
    }
    adapter = ExternalCommandAdapter(spec)
    assert adapter.availability().available
    command = adapter.dry_run_command(ds, 2, 0, str(tmp_path))
    assert stub in command and "counts.tsv" in command

    run = adapter.run(ds, n_states=2, seed=0, workdir=str(tmp_path / "work"),
                      timeout_s=120.0)
    assert run.status == "ok", run.reason
    assert run.coverage == pytest.approx(1.0)
    assert set(np.unique(run.states)) <= {0, 1}
    assert not run.verified, "an unverified spec must stay flagged through the run"

    table = evaluate_runs([run], ds, truth=truth)
    assert table["ari"].notna().all()
    assert table["states_used"].iloc[0] <= 2


def test_external_tool_failure_is_reported_not_raised(small, tmp_path):
    ds, _ = small
    spec = {
        "key": "broken",
        "display_name": "Deliberately broken tool",
        "enabled": True,
        "input_format": "count_matrix",
        "command": [sys.executable, "-c", "import sys; sys.exit(3)"],
        "output": {"format": "segmentation_bed", "path": "{outdir}/segmentation.bed"},
    }
    run = ExternalCommandAdapter(spec).run(ds, n_states=3, seed=0,
                                           workdir=str(tmp_path / "w"), timeout_s=60.0)
    assert run.status == "failed"
    assert "exited 3" in run.reason


def test_missing_output_is_a_failure_not_an_empty_segmentation(small, tmp_path):
    ds, _ = small
    spec = {
        "key": "silent",
        "display_name": "Tool that writes nothing",
        "enabled": True,
        "input_format": "count_matrix",
        "command": [sys.executable, "-c", "pass"],
        "output": {"format": "segmentation_bed", "path": "{outdir}/segmentation.bed"},
    }
    run = ExternalCommandAdapter(spec).run(ds, n_states=3, seed=0,
                                           workdir=str(tmp_path / "w"), timeout_s=60.0)
    assert run.status == "failed"
    assert "could not read output" in run.reason


# -- config file -------------------------------------------------------------


def test_shipped_tools_json_is_valid_and_disabled_by_default():
    with open(os.path.join(TOOLS_DIR, "tools.json")) as fh:
        spec = json.load(fh)
    keys = set()
    for tool in spec["tools"]:
        for field in ("key", "display_name", "command", "output", "how_to_enable"):
            assert field in tool, f"{tool.get('key')} is missing {field}"
        assert tool["key"] not in keys
        keys.add(tool["key"])
        assert tool["enabled"] is False, "shipped entries must not run until configured"
        assert tool.get("verified") is False, "nothing here has been run against the real tool"
        assert tool["key"] in registry()


def test_count_matrix_round_trips_through_the_cli_loader(small, tmp_path):
    import pandas as pd
    from toolbench.cli import _dataset_from_frame

    ds, _ = small
    path = export_count_matrix(ds, str(tmp_path / "counts.tsv"))
    back = _dataset_from_frame(pd.read_csv(path, sep="\t"))
    assert back.n_rows == ds.n_rows
    assert back.marks == ds.marks
    np.testing.assert_allclose(np.sort(back.X, axis=0), np.sort(ds.X, axis=0))
    assert sum(sl.stop - sl.start for sl in back.chains) == back.n_rows


# -- real-data evaluation (toolbench.realdata) --------------------------------


def _toy_features(ds):
    import pandas as pd

    rows = []
    bins = ds.bins.loc[ds.sample_mask(ds.samples[0])]
    for chrom, grp in bins.groupby("chrom"):
        starts = grp["start"].to_numpy()
        for i, s in enumerate(starts[20:-20:40]):
            rows.append({"gene_id": f"{chrom}_g{i}", "name": f"{chrom}_g{i}", "chrom": chrom,
                         "start": int(s), "end": int(s) + 3000, "strand": "+" if i % 2 else "-"})
    return cs.FeatureSet(pd.DataFrame(rows))


def test_realdata_runs_round_trip_and_figures(small, tmp_path):
    import pandas as pd

    from toolbench.realdata import (
        expression_prediction,
        load_runs,
        promoter_composition,
        promoter_recovery,
        save_runs,
        state_strengths,
        write_realdata_report,
    )

    ds, truth = small
    runs = run_tools(ds, ["chromstate", "kmeans"], n_states=truth.n_states, seed=0)
    save_runs(runs, str(tmp_path / "runs"))
    back = load_runs(str(tmp_path / "runs"))
    assert [r.tool for r in back] == ["chromstate", "kmeans"]
    assert np.array_equal(back[0].states, runs[0].states)
    assert back[0].posterior is not None and back[0].posterior.shape == runs[0].posterior.shape

    f = _toy_features(ds)
    comp = promoter_composition(ds, back[0], f, flank=1000)
    scols = [c for c in comp.columns if c not in ("gene_id", "sample")]
    assert len(scols) == truth.n_states
    assert np.allclose(comp[scols].sum(axis=1), 1.0, atol=1e-6)
    assert set(comp["sample"]) == set(ds.samples)

    st = state_strengths(ds, back[0])
    assert set(st["class"]) <= {"active", "bivalent", "polycomb", "heterochromatin", "weak", "quiescent"}

    rng = np.random.default_rng(0)
    expr = pd.DataFrame({s: rng.normal(size=len(f)) for s in ds.samples}, index=f.table["gene_id"])
    rec = promoter_recovery(ds, back, f, expr, flank=1000, block_bp=20_000)
    assert {"tss_auprc", "active_tss_auprc"} <= set(rec.columns) and len(rec) == 2
    pred = expression_prediction(ds, back, f, expr, flank=1000, block_bp=20_000)
    assert {"level_rho", "change_rho"} <= set(pred.columns)
    assert pred["method"].iloc[-1].startswith("Raw mark")

    chrom = ds.bins["chrom"].iloc[0]
    written = write_realdata_report(str(tmp_path / "rd"), ds, back, f, expr, 1000,
                                    [(chrom, 0, 20_000, "toy")], "chromstate", None, None, "toy", 0)
    names = {os.path.basename(p) for p in written}
    for need in ("state_profiles.pdf", "state_class_fractions.pdf", "segment_lengths.pdf",
                 "state_overlap.pdf", "promoter_recovery.pdf", "expression_prediction.pdf",
                 "locus_toy.pdf", "REPORT.md", "expression_prediction.tsv"):
        assert need in names, need


def test_segway_adapter_is_registered_and_skips_cleanly(small, monkeypatch):
    from toolbench.methods.segway import SegwayAdapter

    monkeypatch.delenv("SEGWAY_ENV", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    ad = SegwayAdapter()
    assert "segway" in registry()
    assert not ad.availability().available
    ds, truth = small
    run = ad.run(ds, n_states=truth.n_states, seed=0, workdir="/nonexistent/x", timeout_s=1)
    assert run.status == "skipped"
    cmd = ad.dry_run_command(ds, 5, 0, "/w")
    assert "genomedata-load" in cmd and "segway train" in cmd and "--num-labels=5" in cmd

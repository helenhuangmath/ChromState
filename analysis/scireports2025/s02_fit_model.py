"""Step 2: choose the number of states and fit one joint model across TN, TMEM and TEX.

K is chosen by held-out likelihood (whole chromosomes held out) on the training
chromosomes; the final model is then refitted on all of them with adaptive restarts.
Output: ``out/model/model.pkl`` plus readable emission/transition/selection tables.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
from common import TRAIN_CHROMS, ensure_out, load_dataset

import chromstate as cs

CANDIDATES = (6, 8, 10, 12, 15)


def choose_k(sel) -> int:
    """Smallest K that captures 95% of the held-out gain over the smallest candidate.

    Held-out likelihood keeps creeping up with K on genome-scale data, so "the maximum"
    is nearly always the largest candidate. The 95%-of-gain rule picks the elbow.
    """
    ll = sel["heldout_loglik_per_bin"].to_numpy()
    gain = ll - ll[0]
    if gain.max() <= 0:
        return int(sel["n_states"].iloc[0])
    ok = gain >= 0.95 * gain.max()
    return int(sel["n_states"].to_numpy()[np.argmax(ok)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-k", action="store_true",
                    help="take K from an existing k_selection.tsv instead of re-selecting")
    ap.add_argument("--max-restarts", type=int, default=16)
    args = ap.parse_args()

    out = ensure_out("model")
    ds = load_dataset(TRAIN_CHROMS)
    print(f"training rows: {ds.n_rows:,} ({', '.join(TRAIN_CHROMS)} x 3 conditions)", flush=True)

    t0 = time.time()
    if args.reuse_k and (out / "k_selection.tsv").exists():
        import pandas as pd
        sel = pd.read_csv(out / "k_selection.tsv", sep="\t")
    else:
        sel = cs.select_n_states(ds, candidates=CANDIDATES, n_restarts=2, n_iter=60,
                                 holdout_fraction=0.34, seed=1, verbose=True)
        sel.to_csv(out / "k_selection.tsv", sep="\t", index=False)
    k = choose_k(sel)
    print(f"selected K={k}  ({time.time() - t0:.0f}s)", flush=True)

    # Restart until the best optimum has been reached by two independent starts: the
    # likelihood surface has real local optima, and a one-off best is not trustworthy.
    t0 = time.time()
    seg = cs.fit(ds, n_states=k, n_restarts=4, max_restarts=args.max_restarts, seed=11)
    print(f"fit K={k}: loglik {seg.loglik:,.0f}, restarts {seg.meta['n_restarts']}, "
          f"hits at best {seg.meta['restart_hits_at_best']}, collapsed {seg.meta['collapsed']} "
          f"({time.time() - t0:.0f}s)", flush=True)

    seg.emission_frame().to_csv(out / "emissions.tsv", sep="\t")
    seg.transition_frame().to_csv(out / "transitions.tsv", sep="\t")
    seg.state_table().to_csv(out / "states.tsv", sep="\t", index=False)
    seg.sample_state_matrix().to_csv(out / "occupancy_train.tsv", sep="\t")
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"hmm": seg.model, "labels": seg.labels, "colors": seg.colors,
                     "rules": seg.rules, "marks": seg.marks, "meta": seg.meta, "k": k,
                     "occupancy": seg.posterior.mean(axis=0)}, fh)
    print(f"restart log-likelihoods: {sorted(round(x) for x in seg.meta['restart_logliks'])}")
    print(seg.emission_frame().round(1))


if __name__ == "__main__":
    main()

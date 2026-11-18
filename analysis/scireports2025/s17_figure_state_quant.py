"""Step 17: figure for step 16 -- which representation, and which states, explain expression.

a  Held-out Spearman rho (95% bootstrap CI) of each chromatin representation, for
   expression level in TN and the three expression-change contrasts.
b  Signed unique importance of each state: held-out R^2 lost when the state is permuted
   in the full model, signed by the direction of its association with expression.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
from common import OUT, ensure_out

from chromstate.model.annotate import full_state_name

FIG = ensure_out("figures")
RES = OUT / "state_quant"

FAMILIES = [
    ("chromhmm_tss", "ChromHMM state at TSS"),
    ("top_state", "ChromState top state"),
    ("chromhmm_fraction", "ChromHMM composition"),
    ("signal", "Raw signal"),
    ("occupancy", "State occupancy"),
    ("occupancy+entropy+biv (step 5)", "Occupancy + entropy + bivalency"),
    ("quantitative", "Quantitative (+ intensity, dose)"),
    ("quantitative+signal", "Quantitative + raw signal"),
]
TASKS = [
    ("expression level in Naive", "Level\nTN"),
    ("log2 fold change Arm vs Naive", "Change\nTMEM vs TN"),
    ("log2 fold change Cl13 vs Naive", "Change\nTEX vs TN"),
    ("log2 fold change Cl13 vs Arm", "Change\nTEX vs TMEM"),
]
N_STATES_SHOWN = 10


def main() -> None:
    ns.apply()
    scores = pd.read_csv(RES / "scores.tsv", sep="\t")
    imp = pd.read_csv(RES / "importance.tsv", sep="\t")

    fig = plt.figure(figsize=(ns.DOUBLE, 72 * ns.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.55)

    # a: representation comparison
    ax = fig.add_subplot(gs[0, 0])
    width = 0.8 / len(FAMILIES)
    shades = plt.get_cmap("Greys")(np.linspace(0.25, 0.75, len(FAMILIES) - 2))
    colors = list(shades) + [ns.ACCENT, ns.BASELINE_DARK]
    for j, (fam, name) in enumerate(FAMILIES):
        sub = scores.set_index(["task", "family"])
        xs, ys, lo, hi = [], [], [], []
        for i, (task, _) in enumerate(TASKS):
            if (task, fam) not in sub.index:
                continue
            row = sub.loc[(task, fam)]
            xs.append(i - 0.4 + (j + 0.5) * width)
            ys.append(row["spearman"])
            lo.append(row["spearman"] - row["rho_lo"])
            hi.append(row["rho_hi"] - row["spearman"])
        ax.bar(xs, ys, width=width * 0.92, color=colors[j], label=name, linewidth=0)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=ns.INK, elinewidth=0.4,
                    capsize=0)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t[1] for t in TASKS])
    ax.set_ylabel("Held-out Spearman ρ")
    ax.set_ylim(0, 0.9)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.0,
              columnspacing=0.8)
    ns.panel_label(ax, "a", x=-0.12)

    # b: signed unique importance per state
    ax = fig.add_subplot(gs[0, 1])
    imp["signed"] = imp["unique_r2"].clip(lower=0) * np.sign(imp["direction"])
    wide = imp.pivot_table(index="state", columns="task", values="signed")
    wide = wide[[t for t, _ in TASKS if t in wide.columns]]
    top = imp.groupby("state")["unique_r2"].max().sort_values(ascending=False)
    wide = wide.loc[top.index[:N_STATES_SHOWN]]
    vmax = float(np.nanmax(np.abs(wide.to_numpy())))
    im = ax.imshow(wide.to_numpy(), cmap=ns.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(wide.to_numpy()):
        ax.text(c, r, f"{abs(v):.2f}", ha="center", va="center", fontsize=4.5, color=ns.INK)
    ax.set_xticks(range(wide.shape[1]))
    short = {t: lbl.replace("Change\n", "").replace(" vs ", "\nvs ") for t, lbl in TASKS}
    ax.set_xticklabels([short[t] for t in wide.columns])
    ax.set_yticks(range(wide.shape[0]))
    ax.set_yticklabels([full_state_name(s) for s in wide.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Unique R², signed by direction", fontsize=5.5)
    cb.outline.set_linewidth(0.4)
    ns.panel_label(ax, "b", x=-0.75)

    ns.save(fig, FIG, "fig_state_quantification")
    print(f"wrote {FIG / 'fig_state_quantification.pdf'}")


if __name__ == "__main__":
    main()

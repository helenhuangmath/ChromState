# How ChromState works

## Model

The genome is cut into fixed bins (200 bp by default). For sample $s$, mark $m$ and bin
$b$, the read count $y_{smb}$ is modelled as negative binomial given the hidden state
$z_{sb}$:

$$
y_{smb} \mid z_{sb}=k \;\sim\; \mathrm{NB}\!\left(\text{mean}= f_{sm}\,\mu_{km},\;\text{dispersion}= \phi_{m}\right)
$$

- $\mu_{km}$ is the emission mean of mark $m$ in state $k$.
- $f_{sm}$ is a size factor that absorbs sequencing depth (median of ratios by
  default; total depth or spike-in on request).
- The states follow a first-order Markov chain along each chromosome of each sample.
  Chains never cross a chromosome or sample boundary.

Counts are used as they are. They are not binarised, which is the main difference from
ChromHMM, and they are not transformed, which is the main difference from Segway.
Bernoulli emissions (the ChromHMM model) and Gaussian emissions are available with
`--emission` for processed inputs such as peaks and coverage tracks.

## Fitting

Parameters are estimated by Baum-Welch expectation-maximisation.

1. **Start.** Emissions are seeded by k-means on log counts, or from a state definition
   table when `--states` is given.
2. **Restarts.** EM is run from several random starts (5 by default, up to 20) until the
   best log-likelihood has been reached twice, so a single poor local optimum is not
   reported.
3. **Speed.** The forward-backward and Viterbi recursions are compiled with Numba when
   it is installed, and chains run in parallel threads. `--fit-fraction` fits on a
   block subsample of the genome and then decodes all of it.

## What is reported

The primary output is the **posterior probability of every state in every bin**. The
Viterbi path is written as well, as a ChromHMM-style segment BED. From the posterior:

| Quantity | Meaning |
|---|---|
| Composition | Share of a bin, promoter, gene or region in each state |
| Entropy | How mixed the states are; low entropy means a confident call |
| Bivalency | Joint probability of active and Polycomb marks in the same bin |
| Mark attribution | How much each mark contributes to the state call |
| Strength | Signal relative to the state's own expectation |

## Naming the states

Each state gets a Roadmap-style mnemonic (`TssA`, `EnhA`, `ReprPC`, `Het`, `Quies`, ...)
from its emission pattern. To fix the vocabulary in advance, give a pattern table with
`--states`; `--anchor seed` starts EM from it and `--anchor fixed` pins the defined
marks.

## Choosing the number of states

[`chromstate selectk`](selectk.md) fits each candidate number of states on part of the
genome and scores it by the likelihood of held-out chromosomes.

## Fine resolution

[`chromstate run`](run.md) fits at 200 bp and scores at 10 bp without refitting. A
negative-binomial count over 200 bp is the sum of 20 counts over 10 bp with the mean
divided by 20, and the 200 bp transition matrix becomes its 20th matrix root. The fine
pass is one forward-backward run per chromosome.

## Calibration

A posterior of 0.9 should be right nine times in ten. ChromState measures this against
the consensus of the other replicates and corrects it with one temperature parameter
(`chromstate.calibrate`).

## Comparing conditions

For each region and state, the share of the state is compared between conditions with a
moderated t-test across replicates, with Benjamini-Hochberg correction. A region is
called changed only if it also exceeds the noise floor, which is the size of change
seen between replicates of the same condition. With three or more conditions a
moderated F-test is reported. Every test needs at least two replicates per condition.

## Expression

Per-gene state tables (promoter, body, distal) are the inputs of a gradient-boosted
tree model that predicts expression level or change. Accuracy is always measured on
genes from chromosomes that were held out of training. The importance of a state is the
drop in held-out $R^2$ when that state's columns are permuted.

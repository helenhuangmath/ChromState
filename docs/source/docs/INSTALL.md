# Install ChromState

## Requirements

- Python 3.9, 3.10, 3.11 or 3.12
- Linux or macOS
- NumPy, SciPy, pandas, matplotlib and scikit-learn (installed automatically)

Optional packages:

| Package | Needed for | Extra |
|---|---|---|
| `pysam` | BAM input | `bam` |
| `pyBigWig` | bigWig input | `bigwig` |
| `anndata` | single-cell input | `anndata` |
| `numba` | faster forward-backward and Viterbi (used automatically when present) | `fast` |

## Install with pip

The [GitHub release](https://github.com/helenhuangmath/ChromState/releases/tag/v1.1)
provides a Python wheel and source distribution. Install the wheel with pip.

```bash
python -m pip install https://github.com/helenhuangmath/ChromState/releases/download/v1.1/chromstate-1.1.0-py3-none-any.whl
```

To install version 1.1 with optional dependencies from Git:

```bash
python -m pip install "chromstate[all] @ git+https://github.com/helenhuangmath/ChromState.git@v1.1"
```

From a clone, in editable mode:

```bash
git clone https://github.com/helenhuangmath/ChromState.git
cd ChromState
pip install -e ".[all]"
```

`[all]` installs every optional package. Use `pip install -e .` for the core only, or
pick extras, for example `pip install -e ".[bam,fast]"`.

## Install with conda

```bash
git clone https://github.com/helenhuangmath/ChromState.git
cd ChromState
conda env create -f environment.yml
conda activate chromstate
```

## R interface

`chromstatR` calls the Python package through `reticulate`. Install the Python package
first, then:

```r
install.packages("reticulate")
```

```bash
R CMD INSTALL R/chromstatR
```

## Check the installation

```bash
chromstate --version
chromstate demo -o demo_out      # simulated end-to-end run, about 15 s
```

To run the test suite from a clone:

```bash
pip install -e ".[dev]"
pytest -q
```

## Threads

ChromState runs chains (one chromosome of one sample) in parallel threads. Set the
number with the environment variable `CHROMSTATE_THREADS`:

```bash
export CHROMSTATE_THREADS=8
```

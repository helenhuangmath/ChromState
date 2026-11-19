# Contributing to ChromState

Thank you for helping to improve ChromState. This page explains how to report a problem,
propose a change and get a pull request merged.

## Report a bug or ask a question

Open an issue in the
[issue tracker](https://github.com/helenhuangmath/ChromState/issues). A report is much
easier to act on when it includes:

- the ChromState version (`chromstate --version`), Python version and operating system;
- the full command or script, and the complete error message;
- a small input that reproduces the problem, if you can share one. `chromstate demo`
  output is a good starting point when your own data cannot be shared.

## Suggest a feature

Open an issue that describes the analysis you want to do and what is missing. A concrete
example of the input and the output you expect is more useful than a general request.

## Set up a development copy

```bash
git clone https://github.com/helenhuangmath/ChromState.git
cd ChromState
pip install -e ".[all,dev,docs]"
```

## Make a change

1. Create a branch from `main`.
2. Make the change, with a test for it in `tests/`. New statistical methods need a test
   that recovers a simulated ground truth.
3. Run the local checks:

   ```bash
   ruff check src tests
   pytest -q
   ```

4. If the change affects users, update the documentation in `docs/` and add a line to
   `CHANGELOG.md`.
5. Open a pull request against `main` and describe what changed and why.

## Documentation

The site at <https://helenhuangmath.github.io/ChromState/> is built with Sphinx from
`docs/source`. The tutorials are in `docs/tutorials`. To build it locally:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

The command option tables are generated from the command-line parser, so edit the
argument help in `src/chromstate/cli.py`, not the tables.

## Code style

- Python 3.9 compatible, formatted to 100 columns, checked with `ruff`.
- Public functions have a docstring; they appear in the API reference.
- Figures follow the helpers in `chromstate.viz` so that every plot has the same style.

## Code of conduct

By taking part in this project you agree to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are released under the project's [MIT licence](https://github.com/helenhuangmath/ChromState/blob/main/LICENSE).

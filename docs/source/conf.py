"""Sphinx configuration for the ChromState documentation site."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chromstate  # noqa: E402
from chromstate.cli import build_parser  # noqa: E402

project = "ChromState"
author = "Hua Huang"
copyright = "2026, Hua Huang"
release = chromstate.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_mock_imports = ["pysam", "pyBigWig", "anndata", "numba", "xgboost"]
exclude_patterns = ["_generated/*.txt"]
# Pages that include the repository's own Markdown carry links written for GitHub.
suppress_warnings = ["myst.xref_missing", "myst.header", "image.not_readable"]

html_baseurl = "https://helenhuangmath.github.io/ChromState/"
html_theme = "sphinx_rtd_theme"
html_title = f"ChromState {release} documentation"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_context = {
    "display_github": True,
    "github_user": "helenhuangmath",
    "github_repo": "ChromState",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}


def _generate() -> None:
    """Write what the pages include: option tables from the parser, and the figures."""
    out = os.path.join(HERE, "_generated")
    os.makedirs(out, exist_ok=True)
    parser = build_parser()
    parser.formatter_class = lambda prog: __import__("argparse").HelpFormatter(prog, width=96)
    sub = next(a for a in parser._actions if getattr(a, "choices", None) and isinstance(a.choices, dict))
    os.environ["COLUMNS"] = "96"
    for name, p in sub.choices.items():
        with open(os.path.join(out, f"{name}.txt"), "w") as fh:
            fh.write(p.format_help())
    with open(os.path.join(out, "chromstate.txt"), "w") as fh:
        fh.write(parser.format_help())
    figures = os.path.join(ROOT, "docs", "figures", "paper")
    for name in sorted(os.listdir(figures)):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(figures, name), os.path.join(out, name))


_generate()

#!/bin/bash
# Run a command with every external tool of the comparison enabled.
#
#   bash tools/setup/benchrun.sh python tools/run_tools.py check
#
# Puts the chromstate-bench conda env first on PATH (without activating it), and points
# each adapter at what tools/setup/setup_env.sh installed under $BENCH. Override BENCH or
# CHROMSTATE_BENCH_ENV if you installed elsewhere.
BENCH="${BENCH:-$HOME/tools/chromstate-bench}"
ENV="${CHROMSTATE_BENCH_ENV:-$HOME/miniconda3/envs/chromstate-bench}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PATH="$ENV/bin:$ENV/lib/jvm/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export JAVA_HOME="$ENV/lib/jvm"
export R_LIBS_USER="" R_LIBS=""
export PYTHONPATH="$ROOT/src:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export CHROMHMM_JAR="$BENCH/chromhmm/ChromHMM/ChromHMM.jar"
export EPICSEG_R="$BENCH/epicseg.R"
export NORMR_RSCRIPT="$ENV/bin/Rscript"
export DIHMM_BUILD="$BENCH/dihmm_build"
export DIHMM_PYTHON="$ENV/bin/python"
exec "$@"

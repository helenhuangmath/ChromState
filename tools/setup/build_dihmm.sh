#!/bin/bash
# Build the diHMM C++ reimplementation (github.com/ykai16/diHMM) as a Python 3.11
# extension, using only the chromstate-bench conda env (macOS arm64 tested).
#
#   bash tools/setup/build_dihmm.sh            # -> $BENCH/dihmm_build/dihmm_ext.so
#
# Upstream's CMakeLists hard-codes Python 2.7, -lgomp and /usr/lib OpenBLAS, and the
# tarball ships prebuilt Linux .so files plus a stale CMakeCache.txt, so we copy only the
# sources and compile with one command instead. Departures from upstream, all deliberate:
#
# 1. dihmm_emissions_offbyone.patch: Emissions_Dec's constructor loops
#    `for (j = 0; j <= n_bin_states; j++)`, reading column n_bin_states of an
#    n_bin_states-column matrix and writing one past the end of a std::vector. With
#    Armadillo's bounds checks on (the default) this throws "Mat::col(): index out of
#    bounds" before training starts. The extra column is never read afterwards, so the
#    fix changes no value the model uses.
# 2. dihmm_rowvec_each_col.patch: `b.each_col() % emissions` multiplies each column by a
#    *row* vector. Old/unchecked Armadillo read the row's memory as the column it means;
#    current Armadillo rejects the shape. `.t()` makes the intended product explicit.
#    Both bugs are benign only with Armadillo's bounds checks compiled out, which is
#    presumably how upstream was built.
# 3. dihmm_viterbi_backtrace.patch: the Viterbi backtrace in Viterbi.h stores, at
#    position i, the back-pointer (the best state at i - 1) instead of the state at i, so
#    every annotated label is shifted one bin downstream. Checked against the model's own
#    emissions: on a 4 x 2000-bin simulation with saturated (0/1) emission entries,
#    2128 of 8000 upstream labels gave their bin zero probability; after the fix, 0.
# 4. Bounds checks: compiled out by default (-DARMA_NO_DEBUG), matching upstream speed.
#    DIHMM_BOUNDS_CHECK=1 keeps them; the harness adapter was validated with them on.
# 5. No -fopenmp: Forward_Backward::update_model_parameters is an `omp parallel for`
#    whose body does interm_ms.push_back(...) on a shared std::vector with the
#    `#pragma omp critical` commented out -- a data race. Built serially the pragmas are
#    ignored and training is correct but single-threaded. Annotation stays parallel
#    across chromosomes (std::async in Viterbi.h).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$HOME/tools/chromstate-bench}"
ENV="${CHROMSTATE_BENCH_ENV:-$HOME/miniconda3/envs/chromstate-bench}"
SRC="$BENCH/src/dihmm_cpp"
OUT="$BENCH/dihmm_build"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$SRC"/*.cpp "$SRC"/*.h "$SRC"/*.py "$OUT"/
rm -f "$OUT"/*.pyc
cd "$OUT"
patch -p0 < "$HERE/dihmm_emissions_offbyone.patch"
patch -p0 < "$HERE/dihmm_rowvec_each_col.patch"
patch -p0 < "$HERE/dihmm_viterbi_backtrace.patch"
ARMA_FLAGS="-DARMA_NO_DEBUG"
if [ "${DIHMM_BOUNDS_CHECK:-0}" = "1" ]; then ARMA_FLAGS=""; fi
PYINC="$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
NPINC="$("$ENV/bin/python" -c 'import numpy; print(numpy.get_include())')"
# macOS: the env's clang++ with lazy symbol lookup; Linux: the env's conda g++ (or system g++).
if [ "$(uname)" = "Darwin" ]; then
  CXX="$ENV/bin/clang++"; LINKFLAGS="-undefined dynamic_lookup"
else
  CXX="$(ls "$ENV"/bin/x86_64-conda-linux-gnu-g++ 2>/dev/null || command -v g++)"; LINKFLAGS=""
fi
PYLIB="$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY").replace("lib","",1).split(".")[0])')"
"$CXX" -std=c++14 -O3 -fPIC -shared $LINKFLAGS \
  -DARMA_DONT_USE_WRAPPER $ARMA_FLAGS \
  -I"$ENV/include" -I"$PYINC" -I"$NPINC" \
  Model.cpp Emissions.cpp Emissions_Dec.cpp Forward_Backward.cpp dihmm_ext.cpp \
  -L"$ENV/lib" -Wl,-rpath,"$ENV/lib" \
  -lboost_python311 -lboost_numpy311 -lboost_filesystem -lopenblas \
  $( [ "$(uname)" = "Darwin" ] || echo "-l$PYLIB" ) \
  -o dihmm_ext.so
echo "built $OUT/dihmm_ext.so"

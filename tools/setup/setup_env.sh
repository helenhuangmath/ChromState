#!/bin/bash
# Recreate the isolated environment the external-tool comparison was run in.
#
#   bash tools/setup/setup_env.sh          # ~15-25 min, mostly conda + R source builds
#
# Installs everything into one conda env (default ~/miniconda3/envs/chromstate-bench) and
# downloads/builds the tools under $BENCH (default ~/tools/chromstate-bench). Nothing goes
# into the base env, the system R library, or the user's R library: R packages that have
# no osx-arm64 conda build (bamsignals, normr) and the GitHub-only ones (kfoots, EpiCSeg)
# are compiled into the env's own R library, and install_r_tools.R refuses any other.
#
# Tested on macOS arm64 (Apple silicon), conda 23 + mamba 1.5. Versions that ran:
# ChromHMM 1.28 (OpenJDK 17.0.18), EpiCSeg 1.0 @383b069 + kfoots 1.0 @4fb7c59 on
# R 4.5.3, normR 1.36.0 (Bioconductor 3.22 source), diHMM ykai16@c5a4bd9.
#
# Honest note: the env was built in several increments while finding out what each tool
# needed; the single `create` below is the union of those installs and its solve was
# checked with --dry-run, but this script as a whole has not been re-run from scratch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$HOME/tools/chromstate-bench}"
ENV="${CHROMSTATE_BENCH_ENV:-$HOME/miniconda3/envs/chromstate-bench}"
SOLVER="${SOLVER:-mamba}"   # `conda` works too, slower
GH="https://github.com"
CODELOAD="https://codeload.github.com"

mkdir -p "$BENCH/src" "$BENCH/chromhmm"

# 1. The conda env -------------------------------------------------------------------
if [ ! -d "$ENV" ]; then
  "$SOLVER" create -y -p "$ENV" -c conda-forge -c bioconda --strict-channel-priority \
    python=3.11 "openjdk=17" "r-base=4.5" \
    bioconductor-rsamtools bioconductor-rhtslib bioconductor-qvalue bioconductor-genomeinfodb \
    bioconductor-genomicranges bioconductor-iranges bioconductor-s4vectors \
    bioconductor-biocgenerics bioconductor-edger bioconductor-rtracklayer \
    bioconductor-genomicalignments bioconductor-biobase \
    r-rcpp r-rcpparmadillo r-optparse r-remotes r-ggplot2 r-data.table r-biocmanager \
    r-xml r-restfulr r-rcolorbrewer r-matrixstats r-rcurl r-matrix \
    cmake make pkg-config compilers armadillo "openblas=*=openmp*" \
    libboost-python-devel libboost-devel \
    numpy pandas scipy scikit-learn matplotlib psutil numba pytest
fi

# 2. Downloads (pinned) ----------------------------------------------------------------
fetch() {  # repo sha dest
  curl -sL -o "$BENCH/src/$3.tar.gz" "$CODELOAD/$1/tar.gz/$2"
  rm -rf "$BENCH/src/$3" && mkdir -p "$BENCH/src/$3"
  tar -xzf "$BENCH/src/$3.tar.gz" -C "$BENCH/src/$3" --strip-components=1
}
curl -sL -o "$BENCH/chromhmm/ChromHMM.zip" "$GH/jernst98/ChromHMM/releases/download/v1.28/ChromHMM.zip"
(cd "$BENCH/chromhmm" && unzip -oq ChromHMM.zip)
fetch lamortenera/epicseg 383b069f0093e84ae6615458f64e31ff2ce0fbf8 epicseg
fetch lamortenera/kfoots 4fb7c5986581d42ce981afc538c5c1031fe1c42a kfoots
fetch ykai16/diHMM c5a4bd9b2d984d1af8c1e9d371651206d952f06c dihmm_cpp

# 3. Patched copies for current R (see each .patch) ------------------------------------
rm -rf "$BENCH/src/kfoots_patched" "$BENCH/src/epicseg_patched"
cp -R "$BENCH/src/kfoots" "$BENCH/src/kfoots_patched"
patch -d "$BENCH/src/kfoots_patched" -p0 < "$HERE/kfoots_r43.patch"
cp -R "$BENCH/src/epicseg" "$BENCH/src/epicseg_patched"
patch -d "$BENCH/src/epicseg_patched" -p0 < "$HERE/epicseg_fcone.patch"

# 4. R packages into the env's library, and the EpiCSeg launcher -----------------------
BENCH="$BENCH" PATH="$ENV/bin:/usr/bin:/bin" R_LIBS_USER="" R_LIBS="" \
  "$ENV/bin/Rscript" "$HERE/install_r_tools.R"

# 5. diHMM ------------------------------------------------------------------------------
BENCH="$BENCH" CHROMSTATE_BENCH_ENV="$ENV" bash "$HERE/build_dihmm.sh"

echo "done. Check with: bash tools/setup/benchrun.sh python tools/run_tools.py check"

#!/usr/bin/env bash
#
# Build KenLM, which provides `lmplz` -- the n-gram estimator `NGramLanguageModel` reads from.
#
# Counting n-grams inside python (`train_internal_lm`, or `NGramLanguageModel.fit_counts`) holds
# every n-gram in a dict, roughly 300-400 bytes per token, so it tops out around a few million
# tokens. `lmplz` does the same job by disk-based merge sort in bounded memory and handles corpora
# three orders of magnitude larger, which is why the NGPU-LM paper builds its models with it.
#
# Usage:
#   ./scripts/install_kenlm.sh
#
# The binary lands in externals/kenlm/build/bin, which is the second place `train_kenlm_lm` looks
# (after PATH), so nothing else is needed. Add it to PATH only to run `lmplz` by hand.

set -e

PROJECT_DIR=$(realpath "$(dirname $0)/..")
cd "$PROJECT_DIR" || exit

mkdir -p "$PROJECT_DIR/externals"
cd "$PROJECT_DIR/externals" || exit

# KenLM needs boost, eigen and the compression libraries. It builds without them but silently drops
# features -- notably reading gzipped input -- so install them first rather than debugging later.
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required and was not found. Install it first:"
  echo "  macOS:  brew install cmake boost eigen"
  echo "  Debian: sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev"
  exit 1
fi

# Pinned to a commit, not a tag: kpu/kenlm publishes no releases -- its only tag is `windows` -- so
# a SHA is the sole stable identifier. This is the tree the Boost patch below was written against
# and verified end to end, so a future master reshuffling that block cannot silently break the build.
KENLM_COMMIT=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5

if [ ! -d kenlm ]; then
  # `git clone --depth 1` cannot take a SHA, only a branch or tag, hence fetching it by hand. Still
  # a single-commit download, so it costs no more than the shallow clone it replaces.
  mkdir -p kenlm
  cd kenlm || exit
  git init -q
  git remote add origin https://github.com/kpu/kenlm.git
  git fetch -q --depth 1 origin "$KENLM_COMMIT"
  git checkout -q FETCH_HEAD
  cd "$PROJECT_DIR/externals" || exit
  echo "Cloned kenlm at $KENLM_COMMIT"
else
  # An existing checkout is never touched: it may be deliberately modified, and the Boost patch
  # below edits CMakeLists.txt, so fetching over it would throw that away. Say what is there and
  # let the caller decide.
  CURRENT=$(git -C kenlm rev-parse HEAD 2>/dev/null || echo unknown)
  if [ "$CURRENT" != "$KENLM_COMMIT" ]; then
    echo "NOTE: externals/kenlm is at $CURRENT, not the pinned $KENLM_COMMIT."
    echo "      Building it as-is. To move to the pin: rm -rf externals/kenlm && ./scripts/install_kenlm.sh"
  fi
fi

# KenLM asks Boost for a `system` component. Boost.System has been header-only since 1.69 and
# Boost 1.90 no longer ships a `boost_system` CMake target at all, so the request fails outright:
#
#   CMake Error: Could not find a package configuration file provided by "boost_system"
#
# Dropping it from the component list is the whole fix -- nothing in KenLM needs the compiled
# library, only the headers, which come with Boost either way. Idempotent, so re-running is safe.
python3 - "$PROJECT_DIR/externals/kenlm/CMakeLists.txt" <<'PATCH'
import sys

path = sys.argv[1]
with open(path) as handle:
    source = handle.read()

wanted = """find_package(Boost 1.41.0 REQUIRED COMPONENTS
  program_options
  system
  thread
  unit_test_framework
)"""
if wanted in source:
    with open(path, "w") as handle:
        handle.write(source.replace(wanted, wanted.replace("  system\n", "")))
    print("Patched CMakeLists.txt: dropped the header-only Boost `system` component")
elif "  system\n" not in source:
    print("CMakeLists.txt already patched, or upstream no longer asks for Boost `system`")
else:
    print("WARNING: could not find the Boost component block to patch; if cmake fails on")
    print("         boost_system, remove `system` from find_package(Boost ...) by hand.")
PATCH

rm -rf "$PROJECT_DIR/externals/kenlm/build"
mkdir -p "$PROJECT_DIR/externals/kenlm/build"
cd "$PROJECT_DIR/externals/kenlm/build" || exit

# Two warnings here are expected and harmless. Eigen3 only gates the `interpolate` tool, which
# nothing in this repository uses -- a modern Eigen reports 5.x against KenLM's `3.1.0` request and
# is skipped. The FindBoost deprecation notice is CMake telling KenLM's developers, not you.
cmake ..

if command -v nproc >/dev/null 2>&1; then
  JOBS=$(nproc)
else
  JOBS=$(sysctl -n hw.ncpu)  # macOS has no nproc
fi
make -j "$JOBS"

BIN_DIR="$PROJECT_DIR/externals/kenlm/build/bin"
if [ ! -x "$BIN_DIR/lmplz" ]; then
  echo "Build finished but $BIN_DIR/lmplz is missing -- check the cmake output above."
  exit 1
fi

cd "$PROJECT_DIR" || exit

echo
echo "KenLM built. lmplz is at $BIN_DIR/lmplz"
echo "That is where train_kenlm_lm looks by default, so nothing else is needed:"
echo
echo "    tensorflow_asr train_kenlm_lm --config-path=<config> --datadir=<datadir> --modeldir=<modeldir>"
echo
echo "Run it from the project root, or pass --lmplz=$BIN_DIR/lmplz. To use lmplz directly, add it to PATH:"
echo
echo "    export PATH=\"$BIN_DIR:\$PATH\""
echo

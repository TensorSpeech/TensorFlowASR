#!/usr/bin/env bash
#
# Install the Cloud TPU build of TensorFlow, replacing the stock one.
#
# This is a script rather than a `[project.optional-dependencies]` extra because the two cannot
# be resolved together. `tensorflow` is a base dependency of this project and `tensorflow-tpu`
# ships its own `tensorflow` distribution, so an extra that adds the latter installs *both* --
# resolving `.[tpu]` for linux yields `tensorflow==2.19.1` and `tensorflow-tpu==2.19.1`, and
# whichever lands last wins. Order-dependent, and silent when it goes the wrong way.
#
# So: uninstall, then install. The same thing `setup.sh tpu` did before the move to uv.
#
# `tensorflow-text` keeps working afterwards even though its `tensorflow` requirement is now
# nominally unsatisfied -- the module it imports is still there, provided by `tensorflow-tpu`.
#
# Usage:
#
#   uv sync                                        # project into .venv, with stock TensorFlow
#   ./scripts/install_tpu.sh                       # swap it for the TPU build
#
#   ./scripts/install_tpu.sh --system              # into the system interpreter (Kaggle, Colab)
#   ./scripts/install_tpu.sh --system --python "$(which python3)"
#
# Every argument is forwarded to `uv pip`, so anything it accepts works here.
#
# Re-run this after any `uv sync`: sync is declarative and puts the stock `tensorflow` back.
#
# The version is taken from the `tensorflow` already installed, so it stays in step with whatever
# `uv sync` resolved and cannot drift from the pin in pyproject.toml. Only the minor is pinned
# (`~=2.19.0`, i.e. >=2.19.0 <2.20): the patch levels do not line up between the two
# distributions -- `tensorflow` has a 2.19.0 where `tensorflow-tpu` jumps from 2.19.0rc0 to
# 2.19.1 -- so an exact pin would fail for no good reason. The minor is what has to match, because
# `tensorflow-text` is built against a specific one.
#
# Set TPU_PACKAGE to override the whole requirement, e.g. TPU_PACKAGE=tensorflow-tpu==2.18.0.
#
# On the 2.18 line `libtpu~=2.18.0` existed only on Google's index and needed
#   --find-links https://storage.googleapis.com/libtpu-tf-releases/index.html
# (which is what `setup.sh` passed). From 2.19 the dependency is `libtpu==0.0.10`, which is on
# PyPI, so no extra index is needed. Pass the flag through if you pin back.

set -euo pipefail

PROJECT_DIR=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_DIR" || exit 1

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it with:  pip install uv" >&2
    exit 1
fi

# $1 is the package; everything after it is the uv environment flags, which have to be forwarded
# so the lookup targets the same interpreter the install will.
#
# `|| true` matters: `uv pip show` exits non-zero for a package that is not installed, and with
# `set -e` and `pipefail` that would kill the script inside the command substitution -- before the
# "neither is installed" message below could explain what went wrong. Not found is a normal answer
# here, so it is reported as empty output rather than as a failure.
installed_version() {
    local package=$1
    shift
    uv pip show "$@" "$package" 2>/dev/null | awk '/^Version:/ { print $2; exit }' || true
}

if [ -z "${TPU_PACKAGE:-}" ]; then
    source_package="tensorflow"
    version=$(installed_version tensorflow "$@")

    if [ -z "$version" ]; then
        # Re-run after a successful swap: `tensorflow` is gone and `tensorflow-tpu` is what is
        # there. Reading the version back off it keeps the script idempotent.
        source_package="tensorflow-tpu"
        version=$(installed_version tensorflow-tpu "$@")
    fi

    if [ -z "$version" ]; then
        echo "Neither tensorflow nor tensorflow-tpu is installed, so there is no version to match." >&2
        echo "Install the project first (uv sync), or set TPU_PACKAGE explicitly." >&2
        exit 1
    fi

    minor=$(printf '%s' "$version" | awk -F. '{ print $1 "." $2 }')
    TPU_PACKAGE="tensorflow-tpu~=${minor}.0"
    echo "==> Found $source_package $version, matching it with $TPU_PACKAGE"
fi

echo "==> Removing the stock TensorFlow (it cannot coexist with $TPU_PACKAGE)"
uv pip uninstall "$@" tensorflow

echo "==> Installing $TPU_PACKAGE"
uv pip install "$@" "$TPU_PACKAGE"

# Confirm what is actually installed. `uv pip list` takes the same environment flags, so "$@"
# still applies; `uv run` would not, since it has no --system.
#
# `|| true` because `set -o pipefail` is on and grep exits 1 when it matches nothing, which would
# report a failure for an install that actually succeeded.
echo "==> Installed TensorFlow distributions:"
uv pip list "$@" | grep -iE "^(tensorflow|libtpu)" | sed 's/^/    /' || true

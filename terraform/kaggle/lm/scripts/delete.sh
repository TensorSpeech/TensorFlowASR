#!/usr/bin/env bash
#
# Delete the kernel. Runs from the destroy-time provisioner, which can only pass values that
# were stored in the resource, so credentials come from the file written at apply time.

set -euo pipefail

: "${KERNEL_ID:?KERNEL_ID is not set}"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "The Kaggle CLI is not on PATH, so $KERNEL_ID cannot be deleted." >&2
    echo "Install it (pip install kaggle) and run:  kaggle kernels delete $KERNEL_ID -y" >&2
    exit 1
fi

# build/kaggle.json is gone on a fresh checkout that inherited someone else's state. Fall back
# to however the CLI normally finds credentials: KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle.
if [ ! -f "${KAGGLE_CONFIG_DIR:-}/kaggle.json" ]; then
    echo "==> No kaggle.json in ${KAGGLE_CONFIG_DIR:-<unset>}; using the ambient Kaggle credentials"
    unset KAGGLE_CONFIG_DIR
fi

echo "==> Deleting $KERNEL_ID"

if output=$(kaggle kernels delete "$KERNEL_ID" -y 2>&1); then
    echo "$output"
    exit 0
fi

echo "$output" >&2

# Already gone is the outcome we wanted, so do not block the destroy on it. Anything else is a
# real failure and should stop, rather than leaving Terraform claiming it removed a live kernel.
if printf '%s' "$output" | grep -qiE '404|not found|does not exist'; then
    echo "==> $KERNEL_ID does not exist on Kaggle; treating as already deleted"
    exit 0
fi

echo "==> Could not delete $KERNEL_ID. Delete it at https://www.kaggle.com/code/$KERNEL_ID" >&2
echo "    then re-run, or drop it from state with: terraform state rm terraform_data.kernel" >&2
exit 1

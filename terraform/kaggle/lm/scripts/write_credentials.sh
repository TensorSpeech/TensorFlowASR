#!/usr/bin/env bash
#
# Write the credentials file the Kaggle CLI reads from KAGGLE_CONFIG_DIR.
#
# Terraform hands the key over through a provisioner `environment` block, which is not
# persisted, so the key never reaches terraform.tfstate. It still has to land on disk because
# `terraform destroy` runs scripts/delete.sh, and destroy-time provisioners are only allowed to
# reference `self` -- they cannot be given `var.kaggle_key`.

set -euo pipefail

: "${KAGGLE_USERNAME:?KAGGLE_USERNAME is not set}"
: "${KAGGLE_KEY:?KAGGLE_KEY is not set}"
: "${KAGGLE_CONFIG_DIR:?KAGGLE_CONFIG_DIR is not set}"

mkdir -p "$KAGGLE_CONFIG_DIR"

# python3 rather than printf: it escapes the values properly instead of trusting that a key
# never contains a quote or a backslash.
python3 - <<'PY'
import json
import os
import stat

path = os.path.join(os.environ["KAGGLE_CONFIG_DIR"], "kaggle.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}, handle)
os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
print("==> Wrote credentials to " + path + " (mode 0600)")
PY

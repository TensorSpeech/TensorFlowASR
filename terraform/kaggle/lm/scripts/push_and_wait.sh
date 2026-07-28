#!/usr/bin/env bash
#
# Push the generated notebook to Kaggle, optionally wait for the run to finish, and download
# whatever it wrote to /kaggle/working.

set -euo pipefail

: "${KERNEL_ID:?KERNEL_ID is not set}"
: "${BUILD_DIR:?BUILD_DIR is not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR is not set}"
: "${WAIT_FOR_COMPLETION:=true}"
: "${POLL_INTERVAL_SECONDS:=60}"
: "${TIMEOUT_MINUTES:=720}"
: "${DOWNLOAD_OUTPUT:=true}"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "The Kaggle CLI is not on PATH. Install it with:  pip install kaggle" >&2
    exit 1
fi

echo "==> Pushing $KERNEL_ID from $BUILD_DIR"
kaggle kernels push -p "$BUILD_DIR"

kernel_url="https://www.kaggle.com/code/$KERNEL_ID"

if [ "$WAIT_FOR_COMPLETION" != "true" ]; then
    echo "==> Pushed. Not waiting. Watch it at $kernel_url"
    exit 0
fi

deadline=$(($(date +%s) + TIMEOUT_MINUTES * 60))
echo "==> Waiting for $KERNEL_ID to finish (checking every ${POLL_INTERVAL_SECONDS}s, giving up after ${TIMEOUT_MINUTES}m)"
echo "==> $kernel_url"

while true; do
    # The CLI prints:  <kernel> has status "COMPLETE"
    # Pull the quoted status out rather than matching the whole line, so a kernel slug that
    # happens to contain "error" or "complete" cannot be mistaken for a status.
    status_line=$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)
    # Not named `status`: that is a read-only variable in zsh, and this file is short enough
    # that someone will eventually run it with the wrong shell.
    kernel_status=$(printf '%s' "$status_line" | sed -n 's/.*has status "\([^"]*\)".*/\1/p' | tr '[:upper:]' '[:lower:]')

    echo "$(date +%H:%M:%S)  ${kernel_status:-$status_line}"

    case "$kernel_status" in
        *complete*)
            echo "==> Finished"
            break
            ;;
        *error*)
            echo "==> The kernel failed. Fetching the log." >&2
            mkdir -p "$OUTPUT_DIR"
            kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR" || true
            echo "==> Log and any output are in $OUTPUT_DIR ; the notebook is at $kernel_url" >&2
            exit 1
            ;;
        *cancel*)
            echo "==> The kernel was cancelled ($kernel_status)" >&2
            exit 1
            ;;
    esac

    # An empty status means the status call itself failed -- a transient network error, or the
    # kernel is not queryable yet just after a push. Keep polling; the timeout is the backstop.

    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "==> Gave up waiting after ${TIMEOUT_MINUTES}m. The kernel is still running on Kaggle:" >&2
        echo "    $kernel_url" >&2
        exit 1
    fi

    sleep "$POLL_INTERVAL_SECONDS"
done

if [ "$DOWNLOAD_OUTPUT" = "true" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "==> Downloading output to $OUTPUT_DIR"
    kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
fi

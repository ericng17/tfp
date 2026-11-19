#!/usr/bin/env bash
# Run on your laptop, not on the GPU server. Pull consistent hourly snapshots.
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
    cat <<'USAGE'
Usage: bash scripts/sync_yolo_checkpoint.sh HOST PORT LOCAL_DIR [REMOTE_DIR] [INTERVAL_SECONDS]

Example (run from the local tfp repository):
  bash scripts/sync_yolo_checkpoint.sh root@SERVER_IP SSH_PORT \
    "$PWD/checkpoints/yolo-roi-v1" /workspace/tfp/checkpoints/yolo-roi-v1 300

Default remote directory: /workspace/tfp/checkpoints/yolo-roi-v1
Default polling interval: 300 seconds; use 0 for a single copy.
Uses your normal SSH authentication. Does not delete existing local files.
USAGE
    exit 2
fi

remote_host=$1
ssh_port=$2
local_dir=$3
remote_dir=${4:-/workspace/tfp/checkpoints/yolo-roi-v1}
interval=${5:-300}

if [[ ! $remote_host =~ ^([[:alnum:]_.-]+@)?[[:alnum:]_.-]+$ || $remote_host == -* ]]; then
    printf 'Invalid SSH host or alias.\n' >&2
    exit 2
fi
if [[ ! $ssh_port =~ ^[0-9]{1,5}$ ]] || (( 10#$ssh_port < 1 || 10#$ssh_port > 65535 )); then
    printf 'PORT must be between 1 and 65535.\n' >&2
    exit 2
fi
if [[ ! $interval =~ ^[0-9]{1,8}$ ]]; then
    printf 'INTERVAL_SECONDS must be a nonnegative integer.\n' >&2
    exit 2
fi
interval=$((10#$interval))
for command_name in ssh rsync; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Missing command: %s\n' "$command_name" >&2
        exit 1
    fi
done
mkdir -p -- "$local_dir"

while true; do
    printf '[%s] Syncing %s:%s → %s\n' "$(date -Is)" "$remote_host" "$remote_dir" "$local_dir"
    sync_status=0
    rsync --archive --partial --delay-updates --protect-args \
        --exclude='.*.tmp' \
        -e "ssh -p $ssh_port -o ConnectTimeout=20" \
        -- "${remote_host}:${remote_dir%/}/" "${local_dir%/}/" || sync_status=$?
    if (( sync_status == 0 )); then
        printf 'Local checkpoint updated.\n'
    else
        printf 'Sync failed (code %s). The first hourly server snapshot may not exist yet.\n' "$sync_status" >&2
    fi
    if (( interval == 0 )); then
        exit "$sync_status"
    fi
    sleep "$interval"
done

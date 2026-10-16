#!/usr/bin/env bash
# Create the symlinks that connect the pipeline scripts to the data.
# Reads paths from scripts/env.sh (copy scripts/env.sh.example first).
#
#   source scripts/env.sh && bash scripts/link_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${DATA_DIR:?source scripts/env.sh first}"

link() {  # link <target> <linkpath>
  if [ -e "$1" ]; then ln -sfn "$1" "$2"; echo "  $2 -> $1"; else echo "  SKIP (missing): $1"; fi
}
echo "== data/ shortcuts"
mkdir -p "$ROOT/data"
[ -n "${DATA_DRIVE_ROOT:-}" ] && link "$DATA_DRIVE_ROOT" "$ROOT/data/drive"
link "$DATA_DIR"                          "$ROOT/data/processed_data"
link "${NEW1_ROOT:-$ROOT/data/local/New1}"   "$ROOT/data/New1"
link "${PROBE_ROOT:-$ROOT/data/local/Probe}" "$ROOT/data/Probe"
echo "== processed_data next to the dynamics scripts (they use relative paths)"
for d in dynamics/training dynamics/msnn dynamics/msnn/src simulation/closed_loop; do
  link "$DATA_DIR" "$ROOT/$d/processed_data"
done
echo "done"

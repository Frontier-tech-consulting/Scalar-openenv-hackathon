#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [[ -z "$PING_URL" ]]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"

printf "[1/3] Pinging %s/reset\n" "$PING_URL"
curl -fsS -X POST "$PING_URL/reset" -H 'Content-Type: application/json' -d '{}' >/dev/null

printf "[2/3] Building Docker image\n"
docker build "$REPO_DIR" >/dev/null

printf "[3/3] Running openenv validate\n"
(
  cd "$REPO_DIR"
  openenv validate
)

printf "All checks passed.\n"

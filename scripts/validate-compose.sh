#!/usr/bin/env bash
# scripts/validate-compose.sh

set -euo pipefail

BASE_FILE="docker-compose.yml"
OVERLAY_FILE="${1:?Usage: $0 <overlay-compose-file>}"

if [ ! -f "$BASE_FILE" ]; then
  echo "FATAL: $BASE_FILE not found in $(pwd)"
  exit 1
fi

if [ ! -f "$OVERLAY_FILE" ]; then
  echo "FATAL: $OVERLAY_FILE not found in $(pwd)"
  exit 1
fi

echo "=== Validating $OVERLAY_FILE against $BASE_FILE ==="

# Production compose expects these variables.
# CI/local validation does not have real image tags,
# so use harmless placeholder values.
export BACKEND_IMAGE="${BACKEND_IMAGE:-validation/backend:latest}"
export FRONTEND_IMAGE="${FRONTEND_IMAGE:-validation/frontend:latest}"
export WORKER_IMAGE="${WORKER_IMAGE:-validation/backend:latest}"

# docker-compose.test.yml is standalone.
if [ "$OVERLAY_FILE" = "docker-compose.test.yml" ]; then
  docker compose -f "$OVERLAY_FILE" config --quiet
  echo "✓ $OVERLAY_FILE is standalone and parses OK"
  exit 0
fi

# First validate the merged Compose configuration itself.
docker compose \
  -f "$BASE_FILE" \
  -f "$OVERLAY_FILE" \
  config --quiet

BASE_SERVICES=$(
  docker compose -f "$BASE_FILE" config --services | sort
)

MERGED_SERVICES=$(
  docker compose \
    -f "$BASE_FILE" \
    -f "$OVERLAY_FILE" \
    config --services | sort
)

ORPHANS=$(comm -13 \
  <(echo "$BASE_SERVICES") \
  <(echo "$MERGED_SERVICES")
)

if [ -n "$ORPHANS" ]; then
  echo "✗ $OVERLAY_FILE defines service(s) not present in $BASE_FILE:"
  echo "$ORPHANS" | sed 's/^/    - /'
  echo ""
  echo "  This usually means a typo'd service key."
  echo "  Compose silently creates a new orphan service."
  echo "  Base services are: $(echo "$BASE_SERVICES" | tr '\n' ' ')"
  exit 1
fi

echo "✓ Compose configuration is valid"
echo "✓ All services in $OVERLAY_FILE match services in $BASE_FILE"
echo "  ($(echo "$MERGED_SERVICES" | wc -l | tr -d ' ') services after merge)"
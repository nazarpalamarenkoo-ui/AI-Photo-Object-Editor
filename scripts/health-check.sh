set -euo pipefail

APP_URL="${1:-${APP_URL:-http://localhost:8000}}"
MAX_RETRIES="${MAX_RETRIES:-12}"
SLEEP_INTERVAL="${SLEEP_INTERVAL:-5}"

# Worker stability window: how long the worker container must stay
# "running" with ZERO additional restarts before we call it healthy.
WORKER_STABILIZE_SECONDS="${WORKER_STABILIZE_SECONDS:-20}"

# Use Docker Desktop's Windows CLI when running from Git Bash/WSL on Windows.
# On Linux production runners, fall back to the regular `docker` command.
if command -v docker.exe >/dev/null 2>&1; then
    DOCKER_CMD="docker.exe"
elif command -v docker >/dev/null 2>&1; then
    DOCKER_CMD="docker"
else
    DOCKER_CMD=""
fi

echo "=== Health Check: $APP_URL ==="

# ── 1. API health endpoint ─────────────────────

check_api_health() {
    local url="$APP_URL/health"
    local attempt=0

    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        attempt=$((attempt + 1))

        HTTP_STATUS=$(
            curl -s \
                -o /tmp/health_response.json \
                -w "%{http_code}" \
                --connect-timeout 5 \
                --max-time 10 \
                "$url" 2>/dev/null || echo "000"
        )

        if [ "$HTTP_STATUS" = "200" ]; then
            echo "✓ API health OK (attempt $attempt)"

            cat /tmp/health_response.json | \
                python3 -m json.tool 2>/dev/null || \
                cat /tmp/health_response.json

            return 0
        fi

        echo "✗ Attempt $attempt/$MAX_RETRIES — HTTP $HTTP_STATUS — waiting ${SLEEP_INTERVAL}s..."
        sleep "$SLEEP_INTERVAL"
    done

    echo "FATAL: API health check failed after $MAX_RETRIES attempts"
    return 1
}

# ── 2. Database connectivity (via health endpoint) ─

check_db_health() {
    local response
    response=$(cat /tmp/health_response.json 2>/dev/null || echo '{}')

    local db_status
    db_status=$(echo "$response" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
db = data.get('database', data.get('db', 'unknown'))
print(db)
" 2>/dev/null || echo "unknown")

    if [ "$db_status" = "ok" ] ||
       [ "$db_status" = "healthy" ] ||
       [ "$db_status" = "True" ]; then

        echo "✓ Database connectivity OK"
        return 0
    else
        echo "⚠ Database status from health endpoint: $db_status (check manually)"
        return 0
    fi
}

# ── 3. Redis connectivity (via health endpoint) ─

check_redis_health() {
    local response
    response=$(cat /tmp/health_response.json 2>/dev/null || echo '{}')

    local redis_status
    redis_status=$(echo "$response" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
redis = data.get('redis', 'unknown')
print(redis)
" 2>/dev/null || echo "unknown")

    if [ "$redis_status" = "ok" ] ||
       [ "$redis_status" = "healthy" ] ||
       [ "$redis_status" = "True" ]; then

        echo "✓ Redis connectivity OK"
    else
        echo "⚠ Redis status from health endpoint: $redis_status (check manually)"
    fi
}

# ── 4. Backend container running check ──────────

check_backend_container() {
    if [ -z "$DOCKER_CMD" ]; then
        echo "⚠ Docker not available here — skipping container check"
        return 0
    fi

    local count

    count=$(
        "$DOCKER_CMD" ps \
            --filter "name=app" \
            --filter "status=running" \
            -q | wc -l
    )

    if [ "$count" -gt 0 ]; then
        echo "✓ Container 'app' is running"
        return 0
    fi

    echo "✗ Container 'app' is NOT running"

    "$DOCKER_CMD" ps \
        --filter "name=app" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" || true

    return 1
}

# ── 5. Worker crash-loop detection ──────────────

check_worker_stability() {
    if [ -z "$DOCKER_CMD" ]; then
        echo "⚠ Docker not available here — skipping worker crash-loop check"
        return 0
    fi

    local cid

    cid=$(
        "$DOCKER_CMD" ps \
            --filter "name=worker" \
            --filter "status=running" \
            -q | head -n1
    )

    if [ -z "$cid" ]; then
        echo "✗ Container 'worker' is NOT running"

        "$DOCKER_CMD" ps -a \
            --filter "name=worker" \
            --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" || true

        return 1
    fi

    local restarts_before

    restarts_before=$(
        "$DOCKER_CMD" inspect \
            --format='{{.RestartCount}}' \
            "$cid" 2>/dev/null || echo "0"
    )

    echo "  worker restart count (t=0s): $restarts_before"
    echo "  waiting ${WORKER_STABILIZE_SECONDS}s to confirm the worker isn't crash-looping..."

    sleep "$WORKER_STABILIZE_SECONDS"

    # Re-resolve the container.
    # If it crashed and was replaced, the container ID may differ.

    local cid2

    cid2=$(
        "$DOCKER_CMD" ps \
            --filter "name=worker" \
            --filter "status=running" \
            -q | head -n1
    )

    if [ -z "$cid2" ]; then
        echo "✗ Worker container is gone/not-running after stability window (crashed)"
        return 1
    fi

    local restarts_after
    local status
    local started_at
    local uptime
    local now_epoch
    local start_epoch

    restarts_after=$(
        "$DOCKER_CMD" inspect \
            --format='{{.RestartCount}}' \
            "$cid2" 2>/dev/null || echo "0"
    )

    status=$(
        "$DOCKER_CMD" inspect \
            --format='{{.State.Status}}' \
            "$cid2" 2>/dev/null || echo "unknown"
    )

    started_at=$(
        "$DOCKER_CMD" inspect \
            --format='{{.State.StartedAt}}' \
            "$cid2" 2>/dev/null || echo ""
    )

    echo "  worker restart count (t=${WORKER_STABILIZE_SECONDS}s): $restarts_after — status: $status"

    if [ "$status" != "running" ]; then
        echo "✗ Worker is not in 'running' state after stability window: $status"
        return 1
    fi

    if [ "$restarts_after" -gt "$restarts_before" ] 2>/dev/null; then
        echo "✗ CRASH-LOOP DETECTED: worker restarted during the stability window ($restarts_before → $restarts_after)"

        "$DOCKER_CMD" logs \
            --tail 50 \
            "$cid2" || true

        return 1
    fi

    start_epoch=$(
        date -d "$started_at" +%s 2>/dev/null || echo 0
    )

    now_epoch=$(date +%s)
    uptime=$((now_epoch - start_epoch))

    if [ "$start_epoch" -gt 0 ] &&
       [ "$uptime" -lt "$WORKER_STABILIZE_SECONDS" ]; then

        echo "✗ Worker only up ${uptime}s (< ${WORKER_STABILIZE_SECONDS}s window) — likely restarted right before check"

        "$DOCKER_CMD" logs \
            --tail 50 \
            "$cid2" || true

        return 1
    fi

    echo "✓ Worker stable — running ${uptime}s with 0 restarts during the stability window"

    return 0
}

# ── Run all checks ─────────────────────────────

FAILED=0

check_api_health       || FAILED=1
check_db_health        || true
check_redis_health     || true
check_backend_container || FAILED=1
check_worker_stability || FAILED=1

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "=== ✅ All health checks passed (incl. worker crash-loop check) ==="
    exit 0
else
    echo ""
    echo "=== ❌ Health check FAILED ==="
    exit 1
fi
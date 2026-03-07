#!/bin/sh
set -e

# entrypoint.sh - handles two deployment scenarios:
#
#   1. LOCAL / docker-compose - mlruns.db is volume-mounted from the host
#      (Windows). Windows artifact paths are patched to Linux-compatible paths.
#
#   2. CLOUD / Render (fresh deploy) - no pre-existing mlruns.db.
#      An empty database is created so MLflow can initialise on first use.
#      No path patching needed (tracking was never done on Windows in the cloud).
#
# In both cases the container works from a writable copy of the DB kept in
# /tmp so the original mount (if any) is never modified.

ORIGINAL_DB="${MLFLOW_DB_PATH:-/app/mlruns.db}"
PATCHED_DB="/tmp/mlruns_patched.db"

if [ -f "$ORIGINAL_DB" ]; then
    # Case 1: DB exists (local volume mount)
    echo "[entrypoint] Found $ORIGINAL_DB - copying to $PATCHED_DB ..."
    cp "$ORIGINAL_DB" "$PATCHED_DB"

    echo "[entrypoint] Patching Windows artifact URIs in the DB ..."
    python3 - <<'PYEOF'
import sqlite3
import re

db = "/tmp/mlruns_patched.db"
conn = sqlite3.connect(db)
c = conn.cursor()

# ------------------------------------------------------------------
# MLflow stores artifact paths in several places:
#
#   1. runs.artifact_uri
#   2. model_versions.source
#   3. experiments.artifact_location
#
# Windows paths must be converted to Linux-compatible paths.
# ------------------------------------------------------------------

def find_win_mlruns_prefix(value):
    """
    Return (old_prefix, new_prefix) if value contains a Windows-style
    mlruns path, otherwise None.

    Handles:
      file:///F:/Github/.../mlruns -> file:///mlruns
      /F:/Github/.../mlruns       -> /mlruns
    """
    if not value:
        return None

    # URI form
    m = re.search(r'(file:///[A-Za-z]:/.*?/mlruns)', value)
    if m:
        return (m.group(1), "file:///mlruns")

    # Bare path form
    m = re.search(r'(/[A-Za-z]:/.*?/mlruns)', value)
    if m:
        return (m.group(1), "/mlruns")

    return None


patched_total = 0

# ---- runs.artifact_uri -------------------------------------------
c.execute("SELECT run_uuid, artifact_uri FROM runs WHERE artifact_uri IS NOT NULL")

for run_uuid, uri in c.fetchall():
    result = find_win_mlruns_prefix(uri)
    if result:
        old, new = result
        new_uri = uri.replace(old, new)

        c.execute(
            "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
            (new_uri, run_uuid)
        )

        print(f"[patch] runs({run_uuid[:8]}...): {old!r} -> {new!r}", flush=True)
        patched_total += 1


# ---- model_versions.source ---------------------------------------
c.execute("SELECT name, version, source FROM model_versions WHERE source IS NOT NULL")

for name, ver, src in c.fetchall():
    result = find_win_mlruns_prefix(src)
    if result:
        old, new = result
        new_src = src.replace(old, new)

        c.execute(
            "UPDATE model_versions SET source=? WHERE name=? AND version=?",
            (new_src, name, ver)
        )

        print(f"[patch] model_versions({name} v{ver}): {old!r} -> {new!r}", flush=True)
        patched_total += 1


# ---- experiments.artifact_location -------------------------------
c.execute("SELECT experiment_id, artifact_location FROM experiments WHERE artifact_location IS NOT NULL")

for exp_id, loc in c.fetchall():
    result = find_win_mlruns_prefix(loc)
    if result:
        old, new = result
        new_loc = loc.replace(old, new)

        c.execute(
            "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
            (new_loc, exp_id)
        )

        print(f"[patch] experiments({exp_id}): {old!r} -> {new!r}", flush=True)
        patched_total += 1


conn.commit()
conn.close()

print(f"[patch] Done - {patched_total} path(s) patched.", flush=True)
PYEOF

else
    # Case 2: No DB (fresh cloud / Render deploy)
    echo "[entrypoint] No mlruns.db found at $ORIGINAL_DB (fresh deploy)."
    echo "[entrypoint] Creating empty MLflow database at $PATCHED_DB ..."

    python3 -c "import sqlite3; sqlite3.connect('$PATCHED_DB').close()"

    echo "[entrypoint] Empty database created."
fi

# Point MLflow at the writable DB
export MLFLOW_TRACKING_URI="sqlite:////${PATCHED_DB}"
echo "[entrypoint] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# Start API server
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000} ..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
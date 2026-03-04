#!/bin/sh
# entrypoint.sh — patch Windows artifact URIs in mlruns.db before server start.
#
# Problem: MLflow stores artifact paths as absolute host-OS paths when an
# experiment is tracked locally on Windows (e.g. file:///F:/Github/fraud-mlops/mlruns/…).
# Those paths are invalid inside a Linux container.
#
# Solution: copy the mounted mlruns.db to a writable temp location and
# rewrite every artifact_uri row so it points at /mlruns (the container-side
# volume mount) instead of the Windows absolute path.  The original file on
# the host is never modified.

set -e

ORIGINAL_DB="${MLFLOW_DB_PATH:-/app/mlruns.db}"
PATCHED_DB="/tmp/mlruns_patched.db"
LINUX_MLRUNS_ROOT="file:///mlruns"

echo "[entrypoint] Copying MLflow DB: $ORIGINAL_DB → $PATCHED_DB"
cp "$ORIGINAL_DB" "$PATCHED_DB"

echo "[entrypoint] Patching Windows artifact URIs in the DB …"
python3 - <<'PYEOF'
import sqlite3, re

db   = "/tmp/mlruns_patched.db"
conn = sqlite3.connect(db)
c    = conn.cursor()

# ------------------------------------------------------------------
# MLflow stores artifact paths in (at least) three different places:
#
#   1. runs.artifact_uri          → "file:///F:/Github/.../mlruns"
#   2. model_versions.source      → "/F:/Github/.../mlruns/..."  ← bare path!
#   3. experiments.artifact_location (optional, may also be present)
#
# The model_versions.source path is what mlflow.pyfunc.load_model()
# actually uses when resolving "models:/Name/Stage", so we MUST patch it.
# ------------------------------------------------------------------

def find_win_mlruns_prefix(value):
    """
    Return (old_prefix, new_prefix) if value contains a Windows-style
    mlruns path, otherwise None.
    Handles both:
      file:///F:/Github/.../mlruns  → file:///mlruns
      /F:/Github/.../mlruns         → /mlruns
    """
    if not value:
        return None
    # URI form: file:///X:/...
    m = re.search(r'(file:///[A-Za-z]:/.*?/mlruns)', value)
    if m:
        return (m.group(1), "file:///mlruns")
    # Bare path form: /X:/... (Windows drive letter, no file:// scheme)
    m = re.search(r'(/[A-Za-z]:/.*?/mlruns)', value)
    if m:
        return (m.group(1), "/mlruns")
    return None

patched_total = 0

# ---- 1. runs.artifact_uri ----------------------------------------
c.execute("SELECT run_uuid, artifact_uri FROM runs WHERE artifact_uri IS NOT NULL")
for run_uuid, uri in c.fetchall():
    result = find_win_mlruns_prefix(uri)
    if result:
        old, new = result
        patched = uri.replace(old, new)
        c.execute("UPDATE runs SET artifact_uri=? WHERE run_uuid=?", (patched, run_uuid))
        print(f"[patch] runs({run_uuid[:8]}…): {old!r} → {new!r}", flush=True)
        patched_total += 1

# ---- 2. model_versions.source ------------------------------------
c.execute("SELECT name, version, source FROM model_versions WHERE source IS NOT NULL")
for name, ver, src in c.fetchall():
    result = find_win_mlruns_prefix(src)
    if result:
        old, new = result
        patched = src.replace(old, new)
        c.execute(
            "UPDATE model_versions SET source=? WHERE name=? AND version=?",
            (patched, name, ver),
        )
        print(f"[patch] model_versions({name} v{ver}): {old!r} → {new!r}", flush=True)
        patched_total += 1

# ---- 3. experiments.artifact_location ----------------------------
c.execute("SELECT experiment_id, artifact_location FROM experiments WHERE artifact_location IS NOT NULL")
for exp_id, loc in c.fetchall():
    result = find_win_mlruns_prefix(loc)
    if result:
        old, new = result
        patched = loc.replace(old, new)
        c.execute(
            "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
            (patched, exp_id),
        )
        print(f"[patch] experiments({exp_id}): {old!r} → {new!r}", flush=True)
        patched_total += 1

conn.commit()
conn.close()
print(f"[patch] Done — {patched_total} path(s) patched.", flush=True)
PYEOF

# Point MLflow at the patched (writable) copy for the life of this container
export MLFLOW_TRACKING_URI="sqlite:////$PATCHED_DB"
echo "[entrypoint] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# Hand off to uvicorn
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000} …"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

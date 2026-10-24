#!/usr/bin/env bash
# Runs inside a RunPod pod (started by cloud.py launch). Sets up LittleKedi at a fixed commit, fetches the dataset
# from a private Hugging Face dataset repo, resumes from any checkpoint already synced for this run, trains, and
# syncs checkpoints + logs to a private Hugging Face model repo every LK_SYNC_MINUTES. Terminates the pod at the end.
#
# Environment (set by cloud.py): LK_REPO LK_REF LK_RUN LK_DATA_REPO LK_DATA_NAME LK_RUNS_REPO LK_TRAIN_ARGS
#   LK_SYNC_MINUTES LK_MAX_HOURS LK_AUTOTERMINATE HF_TOKEN; RunPod provides RUNPOD_POD_ID and a pod-scoped
#   RUNPOD_API_KEY.
set -uo pipefail
mkdir -p /workspace && cd /workspace
exec > >(tee -a /workspace/bootstrap.log) 2>&1
log() { echo "[littlekedi $(date -u +%H:%M:%S)] $*"; }
log "pod ${RUNPOD_POD_ID:-?} starting run ${LK_RUN} at ${LK_REF}"

DONE=/workspace/.littlekedi-done  # container disk survives restarts: marks a run that already finished

terminate() {
  [ "${LK_AUTOTERMINATE:-1}" = "1" ] || { log "auto-terminate off; pod left running"; return; }
  log "terminating pod ${RUNPOD_POD_ID:-?} (pod API key $([ -n "${RUNPOD_API_KEY:-}" ] && echo present || echo MISSING))"
  # The pod-scoped key works with the GraphQL API (what runpodctl uses); the REST call is a fallback.
  curl -fsS -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-}" -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-}\\\"}) }\"}" | tee /dev/stderr \
      | grep -q '"errors"' && log "GraphQL terminate returned errors" || { sleep 60; log "terminate requested"; }
  curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:-}" -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    || log "REST terminate failed too: stop it with cloud.py terminate"
}

# The container is restarted whenever this script exits. If the run already finished here, don't redo
# set-up or sync: try to terminate again, then idle rather than loop (cloud.py watch can terminate it).
if [ -f "$DONE" ]; then
  log "run ${LK_RUN} already finished on this pod"
  terminate
  log "still running after terminate attempts; idling (terminate with cloud.py)"
  exec sleep infinity
fi

command -v git >/dev/null && command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq git curl >/dev/null; }
[ -d LittleKedi/.git ] || git clone -q "$LK_REPO" LittleKedi
cd LittleKedi && git fetch -q origin && git checkout -q "$LK_REF" || { log "cannot check out $LK_REF"; terminate; exit 1; }
pip install -q --root-user-action=ignore numpy pyarrow "tokenizers>=0.23.2" "huggingface-hub>=1.33" safetensors gguf
python -c "import torch; print('torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))" || { terminate; exit 1; }

export LK_DATA_DIR="data/${LK_DATA_NAME:-}"  # RunPod drops empty env vars, so default it under set -u
if [ -n "${LK_DATA_REPO:-}" ]; then
python - <<'EOF' || { log "data download failed"; terminate; exit 1; }
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["LK_DATA_REPO"], repo_type="dataset", local_dir=os.environ["LK_DATA_DIR"])
print("dataset ready:", os.environ["LK_DATA_DIR"], sorted(os.listdir(os.environ["LK_DATA_DIR"])))
EOF
else
  log "no dataset repo: skipping the dataset download"
fi

# Resume from anything already synced for this run (a previous pod that died or hit its time limit).
mkdir -p checkpoints
RESUME=""
python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
run, repo = os.environ["LK_RUN"], os.environ["LK_RUNS_REPO"]
try:
    d = snapshot_download(repo, allow_patterns=[f"{run}/*"], ignore_patterns=["*.tmp"], local_dir="/workspace/synced")
    src = os.path.join(d, run)
    if os.path.isdir(src):
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f), os.path.join("checkpoints", f))
        print("restored", sorted(os.listdir(src)))
except Exception as e:  # first run: nothing synced yet
    print("nothing to restore:", type(e).__name__)
EOF
[ -f "checkpoints/${LK_RUN}.pt" ] && RESUME="--resume" && log "resuming from synced checkpoint"

sync() {
  python - <<'EOF' || log "sync failed (will retry)"
import os
from huggingface_hub import HfApi
run = os.environ["LK_RUN"]
HfApi().upload_folder(repo_id=os.environ["LK_RUNS_REPO"], folder_path="checkpoints", path_in_repo=run,
                      allow_patterns=[f"{run}.*", f"{run}-*"], ignore_patterns=["*.tmp"], commit_message=f"{run}: sync")
EOF
}
( while sleep $(( ${LK_SYNC_MINUTES:-30} * 60 )); do sync; done ) &
SYNC_PID=$!

if [ -n "${LK_CMD:-}" ]; then  # a custom job (e.g. sft.py) instead of a training run
  log "command: ${LK_CMD}"
  timeout --signal=INT "$(( ${LK_MAX_HOURS:-48} * 3600 ))" bash -c "${LK_CMD}" 2>&1 | tee -a "checkpoints/${LK_RUN}.out"
else
  log "train.py --data-dir ${LK_DATA_DIR} ${LK_TRAIN_ARGS} --out checkpoints/${LK_RUN}.pt ${RESUME}"
  # shellcheck disable=SC2086
  timeout --signal=INT "$(( ${LK_MAX_HOURS:-48} * 3600 ))" \
    python -u -W ignore train.py --data-dir "${LK_DATA_DIR}" ${LK_TRAIN_ARGS} --out "checkpoints/${LK_RUN}.pt" ${RESUME} \
    2>&1 | tee -a "checkpoints/${LK_RUN}.out"
fi
STATUS=${PIPESTATUS[0]}
[ "$STATUS" = "124" ] && log "hit the ${LK_MAX_HOURS}h limit; resume by launching the same run again"
log "training exited with status ${STATUS}; final sync"
kill "$SYNC_PID" 2>/dev/null
sync
[ "$STATUS" = "0" ] && touch "$DONE"
terminate
[ -f "$DONE" ] && { log "idling after a finished run (terminate with cloud.py)"; exec sleep infinity; }

"""Train LittleKedi on RunPod from this repo.

How it fits together:
  * code     - each pod clones the public GitHub repo at an exact commit (so commit and push first)
  * data     - a private Hugging Face dataset repo (upload once with `push-data`)
  * outputs  - checkpoints, logs and milestones sync every N minutes to a private Hugging Face model repo
               (default <user>/littlekedi-runs, one folder per run), so a run survives its pod and can be resumed
               by launching it again; `pull` copies them into checkpoints/
  * clean-up - the pod terminates itself when training ends (or after --max-hours)

Credentials, in .env (gitignored; see .env.example) or as environment variables, never on the command line:
  RUNPOD_API_KEY  your RunPod API key (Settings -> API Keys)
  LK_HF_TOKEN     a Hugging Face token the pod uses to read the dataset and write checkpoints. Use a fine-grained
                  token limited to the dataset and runs repos: it is stored in the pod's environment on RunPod.

    python cloud.py push-data data/mix-8k-4b --repo edededdy/littlekedi-data-mix-8k-4b
    python cloud.py launch mini-wide-sparse --data-repo edededdy/littlekedi-data-mix-8k-4b --data-name mix-8k-4b \\
        --train-args "--preset mini-wide-sparse --max-train-tokens 3.2e9 --batch-size 16 --grad-accum 4 --compile"
    python cloud.py list
    python cloud.py logs mini-wide-sparse            # tail of the synced training log
    python cloud.py pull mini-wide-sparse            # checkpoints + logs -> checkpoints/
    python cloud.py terminate <pod-id>
    python cloud.py watch mini-wide-sparse <pod-id>  # terminates the pod after the run ends, if it didn't itself
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
REPO_URL = "https://github.com/EdmundMartin/LittleKedi.git"
RAW_URL = "https://raw.githubusercontent.com/EdmundMartin/LittleKedi"
DEFAULT_IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel"
DEFAULT_RUNS_REPO = "edededdy/littlekedi-runs"
CUDA_VERSIONS = ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"]  # API enum
LAUNCHES = Path("checkpoints/cloud_launches.jsonl")  # local record of what was launched (checkpoints/ is gitignored)


def load_dotenv(path: Path = Path(__file__).resolve().parent / ".env") -> None:
    """KEY=VALUE lines from .env (gitignored) into the environment; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value and key.strip() not in os.environ:
            os.environ[key.strip()] = value


def runpod(method: str, path: str, body: dict | None = None) -> dict | list:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("set RUNPOD_API_KEY (RunPod -> Settings -> API Keys)")
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          # Cloudflare in front of the API blocks urllib's default UA (error 1010)
                                          "User-Agent": "littlekedi-cloud/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path} failed: {e.code} {e.read().decode(errors='replace')[:500]}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def pushed_commit(allow_dirty: bool) -> str:
    """The commit pods will run: must be committed and on GitHub, or they'd run different code."""
    if git("status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        raise SystemExit("uncommitted changes to tracked files: commit and push them first (or --allow-dirty)")
    head = git("rev-parse", "HEAD")
    git("fetch", "-q", "origin")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"commit {head[:8]} isn't on GitHub yet: git push first")
    return head


def hf_token() -> str:
    tok = os.environ.get("LK_HF_TOKEN")
    if not tok:
        raise SystemExit("set LK_HF_TOKEN to a fine-grained Hugging Face token with read access to the dataset repo "
                         "and write access to the runs repo (it is stored in the pod's environment on RunPod)")
    return tok


def cmd_push_data(a) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"uploading {a.dir} -> {a.repo} (private dataset); resumable, re-run if interrupted")
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.dir)
    print("done")


def cmd_launch(a) -> None:
    ref = pushed_commit(a.allow_dirty)
    env = {"LK_REPO": REPO_URL, "LK_REF": ref, "LK_RUN": a.run, "LK_DATA_REPO": a.data_repo,
           "LK_DATA_NAME": a.data_name, "LK_RUNS_REPO": a.runs_repo, "LK_TRAIN_ARGS": a.train_args,
           "LK_SYNC_MINUTES": str(a.sync_minutes), "LK_MAX_HOURS": str(a.max_hours),
           "LK_AUTOTERMINATE": "0" if a.keep_pod else "1", "LK_CMD": a.command or ""}
    if not a.command and not (a.data_repo and a.data_name and a.train_args):
        raise SystemExit("give --data-repo, --data-name and --train-args (a training run), or --command")
    start = (f"(command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl)) && "
             f"curl -fsSL {RAW_URL}/{ref}/cloud/bootstrap.sh -o /tmp/bootstrap.sh && bash /tmp/bootstrap.sh")
    body = {"name": f"littlekedi-{a.run}", "imageName": a.image, "cloudType": a.cloud, "gpuTypeIds": a.gpu,
            "gpuCount": 1, "containerDiskInGb": a.disk, "volumeInGb": 0, "interruptible": a.spot,
            # hosts whose driver can run the image's CUDA; any older one fails at the first CUDA call
            "allowedCudaVersions": [v for v in CUDA_VERSIONS if float(v) >= float(a.min_cuda)],
            "dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": [start], "env": env}
    shown = {**body, "env": {**env, "HF_TOKEN": "<LK_HF_TOKEN>"}}
    if a.dry_run:
        print(json.dumps(shown, indent=2))
        return
    body["env"]["HF_TOKEN"] = hf_token()
    from huggingface_hub import HfApi
    HfApi().create_repo(a.runs_repo, private=True, exist_ok=True)
    pod = runpod("POST", "/pods", body)
    LAUNCHES.parent.mkdir(exist_ok=True)
    with LAUNCHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"pod": pod.get("id"), "run": a.run, "ref": ref, "gpu": (pod.get("gpu") or {}).get("displayName"),
                            "cost_per_hr": pod.get("costPerHr"), "request": shown}) + "\n")
    print(f"launched pod {pod.get('id')} for run {a.run} at {ref[:8]}: {(pod.get('gpu') or {}).get('displayName', a.gpu)}, "
          f"${pod.get('costPerHr', '?')}/hr")
    print(f"  checkpoints + logs sync to https://huggingface.co/{a.runs_repo}/tree/main/{a.run} every {a.sync_minutes} min")
    print(f"  follow: python cloud.py logs {a.run}   |   stop: python cloud.py terminate {pod.get('id')}")


def cmd_list(_a) -> None:
    pods = runpod("GET", "/pods")
    if not pods:
        print("no pods")
    for p in pods:
        print(f"{p['id']:16s} {p.get('desiredStatus', '?'):10s} {p.get('name', ''):32s} "
              f"{(p.get('gpu') or {}).get('displayName', ''):24s} ${p.get('costPerHr', '?')}/hr")


def cmd_terminate(a) -> None:
    runpod("DELETE", f"/pods/{a.pod}")
    print(f"terminated {a.pod} (synced checkpoints stay in the runs repo)")


def _run_finished(run: str, runs_repo: str) -> bool:
    """True once the synced training log has its final 'end' record (the final checkpoint is in the same sync)."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(runs_repo, f"{run}/{run}.log.jsonl", local_dir="checkpoints/.synced", force_download=True)
    except Exception:  # nothing synced yet
        return False
    return any(json.loads(line).get("event") == "end"
               for line in Path(p).read_text(encoding="utf-8-sig").splitlines() if line.strip())


def cmd_watch(a) -> None:
    """Safety net for self-termination: terminate the pod once its run has finished and synced."""
    import time
    ended = None
    print(f"watching pod {a.pod} (run {a.run}); checking every {a.every} min, grace {a.grace} min after the end")
    while True:
        try:
            if not any(p["id"] == a.pod for p in runpod("GET", "/pods")):
                print(f"pod {a.pod} is gone" + (" (deleted itself)" if ended else ""))
                return
            if ended is None and _run_finished(a.run, a.runs_repo):
                ended = time.time()
                print(f"{time.strftime('%H:%M')} run finished and synced; giving the pod {a.grace} min to delete itself")
            if ended is not None and time.time() - ended >= a.grace * 60:
                print(f"{time.strftime('%H:%M')} pod still up: terminating")
                cmd_terminate(a)
                return
        except Exception as e:  # a network blip (DNS, timeout, 5xx) mustn't end the safety net: try again next round
            print(f"{time.strftime('%H:%M')} check failed, retrying in {a.every} min: {type(e).__name__}: {e}", flush=True)
        time.sleep(a.every * 60)


def _download(run: str, runs_repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(runs_repo, allow_patterns=[f"{run}/{p}" for p in patterns],
                                  local_dir=f"checkpoints/.synced")) / run


def cmd_logs(a) -> None:
    f = _download(a.run, a.runs_repo, [f"{a.run}.out"]) / f"{a.run}.out"
    if not f.exists():
        raise SystemExit(f"no log synced for {a.run} yet (start-up takes a few minutes, then it syncs every "
                         f"--sync-minutes); start-up errors are only in the pod's own log in the RunPod console")
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-a.lines:]))


def cmd_pull(a) -> None:
    d = _download(a.run, a.runs_repo, ["*"])
    out = Path("checkpoints")
    for f in sorted(d.iterdir()):
        target = out / f.name
        if target.exists() and not a.overwrite and f.suffix == ".pt":
            print(f"  skip {f.name} (exists; --overwrite to replace)")
            continue
        target.write_bytes(f.read_bytes())
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push-data", help="upload a prepared dataset dir to a private HF dataset repo")
    p.add_argument("dir"); p.add_argument("--repo", required=True)
    p.set_defaults(fn=cmd_push_data)

    p = sub.add_parser("launch", help="start a pod that trains one run")
    p.add_argument("run", help="run name: checkpoints/<run>.pt, and its folder in the runs repo")
    p.add_argument("--data-repo", default="", help="HF dataset repo holding the prepared dataset")
    p.add_argument("--data-name", default="", help="local dataset dir name on the pod (data/<name>)")
    p.add_argument("--train-args", default="", help="train.py arguments except --data-dir/--out/--resume")
    p.add_argument("--command", default=None,
                   help="run this shell command (from the repo root) instead of train.py, e.g. an sft.py run. Files in "
                        "the run's runs-repo folder are restored to checkpoints/ first; checkpoints/<run>.* and "
                        "<run>-* are synced back. Without --data-repo, no dataset is downloaded")
    p.add_argument("--gpu", nargs="+", default=["NVIDIA GeForce RTX 4090"], help="RunPod GPU type ids, in priority order")
    p.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="COMMUNITY")
    p.add_argument("--spot", action="store_true", help="interruptible (cheaper; resumes from the last sync if killed)")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--min-cuda", default="12.8", choices=CUDA_VERSIONS,
                   help="only hosts whose driver supports at least this CUDA (match the image's)")
    p.add_argument("--disk", type=int, default=60, help="container disk in GB (dataset + checkpoints)")
    p.add_argument("--runs-repo", default=DEFAULT_RUNS_REPO)
    p.add_argument("--sync-minutes", type=int, default=30)
    p.add_argument("--max-hours", type=float, default=48, help="safety limit; training stops and the pod ends")
    p.add_argument("--keep-pod", action="store_true", help="don't terminate the pod when training ends")
    p.add_argument("--allow-dirty", action="store_true", help="launch even with uncommitted changes (pods won't see them)")
    p.add_argument("--dry-run", action="store_true", help="print the pod request without launching")
    p.set_defaults(fn=cmd_launch)

    sub.add_parser("list", help="list your pods").set_defaults(fn=cmd_list)
    p = sub.add_parser("terminate", help="terminate a pod"); p.add_argument("pod"); p.set_defaults(fn=cmd_terminate)
    p = sub.add_parser("watch", help="terminate a pod once its run has finished and synced (self-delete safety net)")
    p.add_argument("run"); p.add_argument("pod"); p.add_argument("--runs-repo", default=DEFAULT_RUNS_REPO)
    p.add_argument("--every", type=float, default=5, help="minutes between checks")
    p.add_argument("--grace", type=float, default=10, help="minutes to let the pod delete itself first")
    p.set_defaults(fn=cmd_watch)
    for name, fn, help_ in (("logs", cmd_logs, "tail a run's synced training log"),
                            ("pull", cmd_pull, "copy a run's synced checkpoints and logs into checkpoints/")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("run"); p.add_argument("--runs-repo", default=DEFAULT_RUNS_REPO)
        if name == "logs":
            p.add_argument("--lines", type=int, default=20)
        else:
            p.add_argument("--overwrite", action="store_true")
        p.set_defaults(fn=fn)

    load_dotenv()
    a = ap.parse_args()
    if getattr(a, "max_hours", None):
        a.max_hours = max(1, int(round(a.max_hours)))
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

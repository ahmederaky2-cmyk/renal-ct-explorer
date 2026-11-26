"""Background worker: ``python -m renal_explorer.patient.worker <case folder>``.

Each job is launched as its own detached process so the browser can disconnect. A file lock
in the store root makes jobs run one at a time (GPU/Metal memory), later ones wait queued.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

from .store import PatientCase, PatientStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def launch(case: PatientCase) -> int:
    case.update_status(state="queued", step="Waiting for the worker", error=None)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")])
    )
    with open(case.log_path, "a") as log:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and module
            [sys.executable, "-m", "renal_explorer.patient.worker", str(case.path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            start_new_session=True,
        )
    return process.pid


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def reconcile(store: PatientStore, grace_seconds: float = 60.0) -> None:
    """Mark jobs whose worker died (reboot, crash) as failed so the UI does not hang.

    A freshly launched job has no PID until the worker records its own, so jobs without a PID
    get a grace period instead of being failed immediately.
    """

    import time

    for case in store.cases():
        status = case.status()
        if status.get("state") not in ("queued", "running"):
            continue
        pid = status.get("pid")
        if process_alive(pid):
            continue
        if pid or time.time() - status.get("updated", 0) > grace_seconds:
            case.discard_raw()
            case.update_status(state="failed", error="Worker stopped unexpectedly; run again.")
    # Orphaned raw uploads (crash between steps) never outlive a finished case.
    for case in store.cases():
        if case.status().get("state") in ("done", "failed") and case.raw_dir.exists():
            case.discard_raw()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m renal_explorer.patient.worker <case folder>", file=sys.stderr)
        return 2
    case_path = Path(argv[1]).resolve()
    case = PatientCase(case_path.name, case_path)
    case.update_status(pid=os.getpid())
    lock_path = case_path.parent / ".worker.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # wait for earlier jobs
        from .pipeline import run_and_record

        run_and_record(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

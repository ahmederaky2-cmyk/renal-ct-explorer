from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from renal_explorer.identity import file_signature


def test_same_size_timestamp_preserving_replacement_changes_signature(tmp_path: Path) -> None:
    """Regression (fifth review): same path, size and mtime must not reuse cached results."""

    ct = tmp_path / "imaging.nii.gz"
    ct.write_bytes(b"A" * 4096)
    original = ct.stat()
    first = file_signature(ct)
    time.sleep(0.02)

    # In-place rewrite with different content, then restore the modification time.
    ct.write_bytes(b"B" * 4096)
    os.utime(ct, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert ct.stat().st_size == original.st_size and ct.stat().st_mtime_ns == original.st_mtime_ns
    second = file_signature(ct)
    assert second != first

    # Replacement by copy-with-metadata (cp -p / restore from backup) under the same name.
    backup = tmp_path / "backup.nii.gz"
    backup.write_bytes(b"C" * 4096)
    os.utime(backup, ns=(original.st_atime_ns, original.st_mtime_ns))
    time.sleep(0.02)
    shutil.copy2(backup, ct)
    assert file_signature(ct) not in (first, second)

    assert file_signature(None) is None and file_signature("") is None

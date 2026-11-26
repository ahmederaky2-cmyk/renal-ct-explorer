"""File identity for cache keys."""

from __future__ import annotations

import os
from pathlib import Path


def file_signature(path: str | os.PathLike | None) -> tuple[int, ...] | None:
    """(device, inode, size, mtime_ns, ctime_ns) of a file, or None.

    ``ctime`` is the inode change time: every write or rename updates it, and unlike the
    modification time it cannot be set back (``touch -r``, ``cp -p``, rsync -t). A CT replaced
    with different intensities, the same size and a preserved timestamp therefore still gets
    a new signature, without hashing hundreds of megabytes on every page load.
    """

    if not path:
        return None
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

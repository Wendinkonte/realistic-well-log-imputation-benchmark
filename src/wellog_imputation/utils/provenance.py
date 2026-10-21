"""Run provenance: which commit of this repository produced an output directory."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["git_commit", "write_run_metadata"]

_REPO = Path(__file__).resolve().parents[3]


def git_commit(repo=_REPO):
    """``(commit, dirty)`` of the repository, or ``(None, None)`` outside a git checkout
    (e.g. an unpacked archive)."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                     "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def write_run_metadata(path, **extra):
    """Write ``path`` (JSON) with the git commit, the command line and ``extra``.

    Written to a temporary name and renamed, so a concurrent reader never sees a partial
    file.  Paths are not recorded (the arguments are), so nothing machine-specific beyond
    what the caller passes ends up in the file.
    """
    from .. import __version__
    head, dirty = git_commit()
    meta = dict(git_commit=head, git_dirty=dirty, package_version=__version__,
                script=os.path.basename(sys.argv[0]), argv=sys.argv[1:],
                utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), **extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, path)
    return meta

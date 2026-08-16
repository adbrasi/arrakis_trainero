"""One huggingface_hub transfer, as a child process.

Run as a child for three reasons: cancelling a job kills the process group and
that has to stop the download; the Xet progress bar then streams into the job
log like every other phase; and the command echoed into that log stays one
readable line you can paste into a shell.

    python -m trainero.hf_fetch '<json spec>'

spec: {repo, remote, subdir, dest, stage}
  remote set  -> single file
  remote ""   -> snapshot of `subdir` (or the whole repo when subdir is "")

The destination only ever appears once the transfer finished: everything lands
in `stage` and is moved into place at the end, so an interrupted download can
never be mistaken for a complete one. The token comes from HF_TOKEN in the
environment — never argv, which is world-readable via /proc.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    spec = json.loads(argv[1])
    dest, stage = Path(spec["dest"]), Path(spec["stage"])

    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        import hf_xet  # noqa: F401
    except ImportError:
        print("[hf] hf_xet ausente — transferência HTTP simples", flush=True)

    stage.mkdir(parents=True, exist_ok=True)
    if spec["remote"]:
        got = hf_hub_download(spec["repo"], filename=spec["remote"], local_dir=str(stage))
        os.replace(got, dest)
    else:
        sub = spec["subdir"]
        snapshot_download(spec["repo"], local_dir=str(stage),
                          **({"allow_patterns": [sub + "/*"]} if sub else {}))
        shutil.rmtree(stage / ".cache", ignore_errors=True)
        inner = stage / sub if sub else stage
        if not inner.is_dir() or not any(inner.iterdir()):
            print(f"[hf] {spec['repo']} não tem nada em {sub or '/'}", flush=True)
            return 1
        os.replace(inner, dest)
    shutil.rmtree(stage, ignore_errors=True)
    print("[hf] ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

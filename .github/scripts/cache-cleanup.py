#!/usr/bin/env python3
"""Deletes Actions caches nothing will read again (cache-cleanup.yml). Needs `gh`.

  cache-cleanup.py --ref REF --all          every cache of REF (a closed pull request's)
  cache-cleanup.py --ref REF --superseded   of REF's caches, those whose key family has
                                            a newer key
  --dry-run                                 print, delete nothing

A key's family is the key without its last `-` segment: the hash of its inputs
(Swatinem/rust-cache's lock-file hash, hashFiles(...)) or a run id. The newest key of
a family is kept (every entry with that key: one key can hold an entry per runner OS).
"""
import json
import subprocess
import sys

# Markers that a set of inputs already passed: each stays valid, and they are tiny.
KEEP_PREFIXES = ("engine-cuda-green-",)


def family(key: str) -> str:
    return key.rsplit("-", 1)[0]


def superseded(caches: list[dict]) -> list[dict]:
    """Of one ref's caches, those with a newer key in the same family."""
    newest: dict[str, dict] = {}
    for c in caches:
        f = family(c["key"])
        if f not in newest or c["created_at"] > newest[f]["created_at"]:
            newest[f] = c
    return [
        c
        for c in caches
        if not c["key"].startswith(KEEP_PREFIXES) and c["key"] != newest[family(c["key"])]["key"]
    ]


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    args = sys.argv[1:]
    ref = args[args.index("--ref") + 1]
    dry = "--dry-run" in args
    assert ("--all" in args) != ("--superseded" in args), __doc__
    pages = json.loads(
        gh("api", "--paginate", "--slurp", f"repos/{{owner}}/{{repo}}/actions/caches?per_page=100&ref={ref}")
    )
    caches = [c for page in pages for c in page["actions_caches"]]
    doomed = caches if "--all" in args else superseded(caches)
    total = 0
    for c in doomed:
        total += c["size_in_bytes"]
        print(f"{'would delete' if dry else 'delete'} {c['size_in_bytes'] / 2**20:7.0f} MB  {c['key']}")
        if not dry:
            # Already gone (evicted, or a second run of this job) is fine.
            subprocess.run(["gh", "api", "-X", "DELETE", f"repos/{{owner}}/{{repo}}/actions/caches/{c['id']}"],
                           capture_output=True)
    print(f"{ref}: {len(doomed)} of {len(caches)} caches, {total / 2**20:.0f} MB")


if __name__ == "__main__":
    main()

# OpenSees oracle: pinned versions

| package | version | wheel | sha256 |
|---|---|---|---|
| openseespy | 3.8.0.0 | `openseespy-3.8.0.0-py3-none-any.whl` (meta package) | `ce6a43503b737025f763a87bebacb1cc2fbd82a156982647e9e014a11b85fdbd` |
| openseespylinux | 3.8.0.0 (OpenSees core 3.8.0) | `openseespylinux-3.8.0.0-py3-none-any.whl` (86 251 016 B) | `2307e40213f1d7128a5a5151d5c8beee8244b8b988528a58f5e44394a045ba64` |
| numpy | 2.5.3 | `numpy-2.5.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `b7e18c623bb5c95acb3b3328861272816ba199fb531921c5d6d0b675f1fde9e3` |

- Python: CPython 3.12.3 (`/home/user/oracle-env/venv`, created with uv 0.11.32), Linux x86_64.
- Installed binary `openseespylinux/opensees.so` (254 608 632 B):
  sha256 `b1bfa97ceb8f0f5d3c3c49eb498ac4e9dc57f86971f0f54c03074968a0f830a3`
  (equal to the wheel's RECORD entry `sb-pfOuPD108PEnrSYrE6dxX-Glx8PVMAwdJaKD4MKM`).
  `pipeline.py` re-hashes it on every run and records the result in the provenance file.
- How the hashes were obtained: the PyPI JSON API (`/pypi/<name>/<version>/json`),
  confirmed by downloading both OpenSees wheels and running `sha256sum`, and by comparing
  the installed `opensees.so` with the wheel's RECORD. (uv's cache stores the hash only
  in a binary archive entry, so it was not used as the source.)
- `install.sh` installs `requirements.txt` with `uv pip install --require-hashes`;
  idempotent (second run: "Checked 3 packages"; against the shared venv a dry run
  reports "Would make no changes"). The hashes are for cp312 / Linux x86_64 only.
- scipy (1.18.1, present in the shared venv) is not used by this oracle.

## License

OpenSeesPy is free of charge but **not OSI open source**. OpenSees itself is under the
UC Berkeley (Regents of the University of California) license, and the bundled
`openseespylinux/LICENSE.md` says: *"OpenSeesPy is free for research, education, and
internal use. Commercial redistribution of OpenSeesPy, such as, but not limited to, an
application or cloud-based service that uses import openseespy, requires a license
similar to that required for commercial redistribution of OpenSees.exe."* (contact
Dr. Minjie Zhu, Oregon State University). Using it as an internal validation oracle is
internal use. Do not ship it or call it from a product or service.

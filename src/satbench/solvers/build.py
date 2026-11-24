"""Fetch, verify and build every solver of ``solvers/manifest.toml``.

Standard library only, so it runs inside a bare build container:

    python3 -m satbench.solvers.build fetch --manifest M --cache DIR      (host, with network)
    python3 -m satbench.solvers.build build --manifest M --cache DIR --out DIR   (container, no network)

``build`` writes ``<out>/build-report.json``. It records, per solver, the archive SHA-256, the build
command, the exit status, the duration and the SHA-256 of every executable produced, so the exact
binaries behind each measured runtime can be audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
from pathlib import Path

USER_AGENT = "satbench/0.1 (academic benchmarking study)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load(manifest: Path) -> dict:
    return tomllib.loads(manifest.read_text())


def archive_file(cache: Path, name: str, url: str) -> Path:
    return cache / f"{name}{''.join(Path(url).suffixes[-2:])}"


def fetch(manifest: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, a in load(manifest)["archives"].items():
        dest = archive_file(cache, name, a["url"])
        if not (dest.exists() and sha256(dest) == a["sha256"]):
            req = urllib.request.Request(a["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        got = sha256(dest)
        if got != a["sha256"]:
            dest.unlink()
            sys.exit(f"{name}: SHA-256 mismatch ({got}); refusing to use it")
        print(f"ok  {name}  {got}")


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extract with tarfile's 'data' filter (no absolute paths, '..', devices or setuid bits).

    Symlinks that point outside the archive are skipped instead of aborting. In competition
    archives they are left-overs of the author's ``configure`` run (e.g. ``src/makefile`` pointing
    into their home directory), and the solver's own build script recreates them. Returns the
    skipped member names so the build report can list them.
    """
    skipped: list[str] = []

    def keep_inside(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        try:
            return tarfile.data_filter(member, path)
        except (tarfile.AbsoluteLinkError, tarfile.LinkOutsideDestinationError):
            skipped.append(member.name)
            return None

    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter=keep_inside)
    return skipped


def remove_prebuilt(root: Path) -> list[str]:
    """Delete ELF files (executables, objects, shared libraries) shipped inside the archive.

    Competition archives often contain the authors' own build products (e.g. a ready-made
    ``kissat`` binary or a whole CMake build tree). Removing them guarantees that every measured
    binary was compiled here from source; if a build silently failed, the solver is missing rather
    than replaced by a foreign binary. Vendored tools under ``third-party/`` are kept.
    """
    removed = []
    for p in sorted(root.rglob("*")):
        if "third-party" in p.relative_to(root).parts or not p.is_file() or p.is_symlink():
            continue
        with open(p, "rb") as f:
            if f.read(4) == b"\x7fELF":
                removed.append(str(p.relative_to(root)))
                p.unlink()
    return removed


BUILD_FILE_NAMES = ("makefile", "Makefile", "GNUmakefile", "CMakeLists.txt", "configure")
BUILD_FILE_SUFFIXES = (".sh", ".mk", ".cmake", ".in", ".make")
ARCH_FLAGS = re.compile(rb"-m(arch|tune)=(native|icelake-server)\b")


def make_portable(root: Path, march: str, exempt: tuple[str, ...] = ()) -> list[str]:
    """Replace machine-specific CPU targets in build files with a fixed, portable ISA level.

    Some solvers build with ``-march=native`` (the binary then depends on the build machine) or
    ``-march=icelake-server`` (AVX-512, which crashes with SIGILL on AMD Zen 3 and older Intel).
    Every such flag becomes ``-march=<march>`` / ``-mtune=generic``, so all binaries run on any
    x86-64 CPU of that level. Solvers that set no CPU target are left untouched. Returns the
    patched files, recorded in the build report.
    """
    patched = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if p.name not in BUILD_FILE_NAMES and not p.name.endswith(BUILD_FILE_SUFFIXES):
            continue
        if str(p.relative_to(root)) in exempt:  # code that hard-requires the original ISA
            continue
        data = p.read_bytes()
        new = ARCH_FLAGS.sub(
            lambda m: b"-march=" + march.encode() if m.group(1) == b"arch" else b"-mtune=generic",
            data,
        )
        if new != data:
            p.write_bytes(new)
            patched.append(str(p.relative_to(root)))
    return patched


def executables(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink() and os.access(p, os.X_OK):
            with open(p, "rb") as f:
                if f.read(4) == b"\x7fELF":
                    out[str(p.relative_to(root))] = sha256(p)
    return out


def build(manifest: Path, cache: Path, out: Path, only: list[str] | None, jobs: int) -> int:
    m = load(manifest)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".work"
    report = {}
    skipped_links: dict[str, list[str]] = {}
    failures = 0
    env = {**os.environ, "MAKEFLAGS": f"-j{jobs}"}
    for name, s in m["solvers"].items():
        if only and name not in only:
            continue
        a = m["archives"][s["archive"]]
        arch = archive_file(cache, s["archive"], a["url"])
        if sha256(arch) != a["sha256"]:
            sys.exit(f"{s['archive']}: archive does not match manifest SHA-256")
        unpack = work / s["archive"]
        if not unpack.exists():
            skipped_links[s["archive"]] = safe_extract(arch, unpack)
        dest = out / name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(unpack / s["path"], dest, symlinks=True)
        prebuilt = remove_prebuilt(dest)
        march = m.get("build", {}).get("portable_march")
        exempt = tuple(s.get("portable_exempt", []))
        portable = make_portable(dest, march, exempt) if march else []
        if s.get("run_sh"):  # for plain releases that ship no competition run script
            (dest / "run.sh").write_text(s["run_sh"])
            (dest / "run.sh").chmod(0o755)
        cmd = s.get("build", "bash build.sh")
        if s.get("pre_build"):
            cmd = f"{s['pre_build']} && {cmd}"
        t0 = time.time()
        with open(out / f"{name}.build.log", "wb") as log:
            rc = subprocess.call(["bash", "-c", cmd], cwd=dest, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
        entry = dict(
            archive=s["archive"], archive_sha256=a["sha256"], path=s["path"], mode=s["mode"],
            ai=s.get("ai"), base=s.get("base"), build=cmd, returncode=rc,
            seconds=round(time.time() - t0, 1), executables=executables(dest),
            skipped_external_links=skipped_links.get(s["archive"], []),
            removed_prebuilt_elf=len(prebuilt),
            portable_march=march, arch_flag_files_patched=portable,
            portable_exempt=list(exempt),
        )
        if s["mode"] == "binary":
            entry["binary"] = s["binary"]
            ok = rc == 0 and (dest / s["binary"]).exists()
        else:
            ok = rc == 0 and (dest / "run.sh").exists() and bool(entry["executables"])
        entry["ok"] = ok
        failures += not ok
        report[name] = entry
        print(f"{'ok ' if ok else 'FAIL'} {name}  rc={rc}  {entry['seconds']}s  "
              f"{len(entry['executables'])} ELF files", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    previous = {}
    if (out / "build-report.json").exists():
        previous = json.loads((out / "build-report.json").read_text())
    previous.update(report)
    (out / "build-report.json").write_text(json.dumps(previous, indent=2) + "\n")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["fetch", "build"])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.action == "fetch":
        fetch(args.manifest, args.cache)
    else:
        sys.exit(1 if build(args.manifest, args.cache, args.out, args.only, args.jobs) else 0)


if __name__ == "__main__":
    main()

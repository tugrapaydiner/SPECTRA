"""Select the current compiler-free wheel, bind it to this checkout, and test it installed.

Use a clean build directory. No version literal or shell wildcard is required.
The wheel is installed into a fresh dependency-free venv and exercised from a
directory outside the checkout.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectra import __version__

REQUIRED_SOURCES = {
    "spectra/_native/exact_binary.cpp", "spectra/_native/quotient_query.cpp",
    "spectra/_native/coloring.cpp", "spectra/_native/sparse_cover.cpp",
}


def select_wheel(directory: Path, expected_version: str = __version__) -> Path:
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--dist must be a directory")
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel; found {len(wheels)}; use a clean build directory")
    wheel = wheels[0]
    if wheel.name != f"spectra-{expected_version}-py3-none-any.whl":
        raise ValueError("expected the current compiler-free SPECTRA wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for entry in archive.infolist():
            raw_name = entry.orig_filename  # ZipInfo normalizes Windows separators and truncates NULs.
            path = PurePosixPath(raw_name)
            if (raw_name != entry.filename or path.is_absolute() or ".." in path.parts
                    or "\\" in raw_name or path.as_posix() != raw_name.rstrip("/")
                    or not path.parts or ":" in path.parts[0]
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("unsafe or linked wheel member")
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("duplicate wheel members or CRC failure")
        metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must contain exactly one distribution metadata file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if metadata.get_all("Name") != ["spectra"] or metadata.get_all("Version") != [expected_version]:
            raise ValueError("wheel name/version metadata does not match the source")
        if not REQUIRED_SOURCES <= set(names):
            raise ValueError("wheel is missing native sources")
    return wheel


SOURCE_PACKAGES = ("spectra",)


def source_files(source_root: Path) -> dict:
    expected = {}
    for package in SOURCE_PACKAGES:
        folder = source_root / package
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"missing source package: {package}")
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".hpp"}:
                if path.is_symlink():
                    raise ValueError("symlinked package source")
                expected[path.relative_to(source_root).as_posix()] = path.read_bytes()
    return expected


def verify_sources(wheel: Path, source_root: Path = ROOT) -> dict:
    """Bind every shipped source file to this checkout, not only hot-path files.

    This is a pure-wheel contract. Native binary distributions need a separate
    build/ABI provenance check and are deliberately not accepted here.
    """
    expected = source_files(source_root)
    with zipfile.ZipFile(wheel) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel source member")
        shipped = {name for name in names if not name.split("/", 1)[0].endswith(".dist-info")}
        if shipped != set(expected):
            raise ValueError(f"wheel/source inventory differs: missing={sorted(set(expected)-shipped)}, extra={sorted(shipped-set(expected))}")
        for name, content in expected.items():
            if archive.read(name) != content:
                raise ValueError(f"wheel/source bytes differ: {name}")
    return {"matched_source_files": len(expected), "source_identity": "PASS"}


def verify_sdist(path: Path, source_root: Path = ROOT, expected_version: str = __version__) -> dict:
    """Check a local build archive without extracting or executing its contents."""
    prefix = f"spectra-{expected_version}"
    if path.name != prefix + ".tar.gz":
        raise ValueError("expected the current SPECTRA source archive")
    expected = source_files(source_root)
    selected = [source_root / name for name in ("MANIFEST.in", "pyproject.toml")]
    selected.extend((source_root / "examples").rglob("*.cnf"))
    # Public documents and figures are explicitly selected by MANIFEST.in.
    for line in (source_root / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "include":
            for pattern in fields[1:]:
                selected.extend(source_root.glob(pattern))
    for item in selected:
        if item.is_symlink() or not item.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("linked or escaping distribution source")
        if not item.is_file():
            raise ValueError("distribution source must be a file")
        expected[item.relative_to(source_root).as_posix()] = item.read_bytes()
    generated = {"PKG-INFO", "setup.cfg"} | {
        "spectra.egg-info/" + name for name in (
            "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
            "requires.txt", "top_level.txt", "zip-safe", "not-zip-safe",
        )
    }
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or name.as_posix() != member.name.rstrip("/")
                    or not name.parts or name.parts[0] != prefix
                    or not (member.isdir() or member.isfile())):
                raise ValueError("unsafe or linked source archive member")
            if member.isdir():
                continue
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if relative in seen:
                raise ValueError("duplicate source archive member")
            seen.add(relative)
            if relative not in expected and relative not in generated:
                raise ValueError(f"unselected source archive member: {relative}")
            limit = len(expected[relative]) if relative in expected else 1024 * 1024
            if member.size > limit:
                raise ValueError(f"source archive member exceeds expected size: {relative}")
            with archive.extractfile(member) as stream:
                raw = stream.read(limit + 1)
            if relative in expected and raw != expected[relative]:
                raise ValueError(f"source archive bytes differ: {relative}")
            if relative.endswith("PKG-INFO"):
                metadata = Parser().parsestr(raw.decode("utf-8"))
                if (metadata.get_all("Name") != ["spectra"]
                        or metadata.get_all("Version") != [expected_version]):
                    raise ValueError("source archive metadata does not match the source")
    if not set(expected) <= seen or "PKG-INFO" not in seen:
        raise ValueError(f"source archive is incomplete: {sorted((set(expected) | {'PKG-INFO'}) - seen)}")
    return {"matched_sdist_files": len(expected), "sdist_identity": "PASS"}


def check_base_installation(wheel: Path, out: Path) -> dict:
    """Install the wheel into a fresh dependency-free venv and run the CNF CLI."""
    wheel = wheel.resolve(strict=True)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as z:
        if z.testzip() is not None:
            raise ValueError('wheel CRC failure')
        natives = sorted(p for p in z.namelist() if p.endswith('.cpp'))
        required = {'spectra/_native/exact_binary.cpp', 'spectra/_native/quotient_query.cpp'}
        if not required <= set(natives):
            raise ValueError('wheel is missing native sources')
    env_dir = out / 'venv'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_dir)
    bindir = env_dir / ('Scripts' if os.name == 'nt' else 'bin')
    python = bindir / ('python.exe' if os.name == 'nt' else 'python')
    command = bindir / ('spectra.exe' if os.name == 'nt' else 'spectra')
    # No source tree or user site can satisfy the installed import by accident.
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    cwd = out / 'outside_checkout'; cwd.mkdir()
    calls = []
    def run(args, expected=0):
        p = subprocess.run([str(a) for a in args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        calls.append(dict(argv=[str(a) for a in args], exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
        (out / 'calls.json').write_text(json.dumps(calls, indent=2)+'\n')
        if p.returncode != expected:
            raise RuntimeError(f'installed command failed: {args}\n{p.stdout}\n{p.stderr}')
        return p.stdout
    run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', wheel])
    probe = run([python, '-I', '-c', '''import importlib.util,json,pathlib,sys
import spectra
from spectra.cnf import CNF,solve,CompactCNFRepairState
assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("numpy") is None
assert "torch" not in sys.modules and "numpy" not in sys.modules
assert pathlib.Path(spectra.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert (pathlib.Path(spectra.__file__).parent/"_native"/"exact_binary.cpp").is_file()
for name in ("common","model","train","eval","data","deploy"):
    assert importlib.util.find_spec(name) is None, name
r=solve(CNF(1,((1,),)),seed=7,max_flips=128)
assert r.status=="SAT_VERIFIED" and r.witness==(True,)
print(json.dumps({"package":spectra.__file__,"version":spectra.__version__,"optional_imports":False}))
'''])
    run([command, '--version'])
    doctor = json.loads(run([command, 'doctor']))
    assert doctor['torch_installed'] is False
    formula = cwd/'input.cnf';formula.write_text('p cnf 1 1\n1 0\n')
    run([command, 'cnf', 'solve', formula, '--seed', '7', '--max-flips', '128', '--out', 'answer.json'])
    answer = json.loads((cwd/'answer.json').read_text());assert answer['status']=='SAT_VERIFIED'
    assert json.loads(run([python, '-I', '-m', 'spectra', 'cnf', 'check', formula, 'answer.json']))['valid']
    run([command, 'cnf', 'solve', formula, '--out', 'answer.json'], expected=2)
    (cwd/'false.json').write_text('{"witness":[false]}')
    run([command, 'cnf', 'check', formula, 'false.json'], expected=1)
    report = dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  native_sources=natives, checks=len(calls), installed_probe=json.loads(probe),
                  status='PASS', mode='fresh venv, no dependencies, outside checkout')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def verify(wheel: Path, out: Path) -> dict:
    """Run the base installed-wheel checks, then the indexed/prepared CNF paths."""
    original = check_base_installation(wheel, out)
    out = out.resolve()
    bindir = out/'venv'/('Scripts' if os.name == 'nt' else 'bin')
    python = bindir/('python.exe' if os.name == 'nt' else 'python')
    cwd = out/'outside_checkout'
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    probe = '''import importlib.util,pathlib,sys
from spectra.cnf import CNF,PreparedCNF,solve,solve_indexed
import spectra.cnf.indexed
assert pathlib.Path(spectra.cnf.indexed.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('numpy') is None
assert 'torch' not in sys.modules
p=CNF(3,((1,2),(-1,3),(-2,3),(1,-1),(3,3)))
a=solve(p,seed=77,max_flips=1024)
for b in [solve_indexed(p,seed=77,max_flips=1024),PreparedCNF(p).solve(seed=77,max_flips=1024)]:
 assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
assert p.satisfied(a.witness)
print('installed indexed/prepared paths match the historical implementation')
'''
    commands = [[python, '-I', '-c', probe],
                [python, '-I', '-m', 'spectra', 'cnf', 'solve', 'input.cnf', '--backend', 'indexed', '--out', 'indexed.json'],
                [python, '-I', '-m', 'spectra', 'cnf', 'check', 'input.cnf', 'indexed.json']]
    calls = []
    for command in commands:
        p = subprocess.run([str(a) for a in command], cwd=cwd, env=env,
                           text=True, capture_output=True, timeout=120)
        calls.append(dict(argv=[str(a) for a in command], exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
        (out/'indexed_calls.json').write_text(json.dumps(calls, indent=2)+'\n')
        if p.returncode != 0:
            raise RuntimeError('installed indexed command failed: '+p.stderr)
    result = dict(status='PASS', original_checks=original['checks'], indexed_checks=len(calls),
                  total_checks=original['checks']+len(calls), scope='actual wheel; clean venv; no Torch/NumPy; outside checkout')
    (out/'indexed_report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-sdist", action="store_true",
                        help="also require and verify the current source archive")
    args = parser.parse_args()
    wheel = select_wheel(args.dist)
    sources = verify_sources(wheel)
    if args.check_sdist:
        archives = sorted(args.dist.glob("*.tar.gz"))
        if len(archives) != 1:
            raise ValueError("expected exactly one source archive; use a clean build directory")
        sources.update(verify_sdist(archives[0]))
    result = {"wheel": str(wheel), "version": __version__, **sources, **verify(wheel, args.out)}
    (args.out / "source_report.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

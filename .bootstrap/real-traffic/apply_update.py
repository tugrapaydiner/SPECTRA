"""Apply the staged real-traffic source update as four readable commits."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "29515edea8d7d4166ad70ac7817ec49eb00698cce9f922826c149403296f45ea"

GROUPS = (
    (
        "Compile the quotient once",
        (
            "experiments/real_traffic/quotient_session.cpp",
            "experiments/real_traffic/quotient_session.py",
            "tests/public/test_real_traffic_quotient_session.py",
        ),
    ),
    (
        "Give MiniCard the same native path",
        (
            "experiments/real_traffic/minicard_session.cpp",
            "experiments/real_traffic/minicard_session.py",
            "tests/public/test_minicard_session.py",
        ),
    ),
    (
        "Check every session in one pass",
        (
            "experiments/real_traffic/session_audit.cpp",
            "experiments/real_traffic/session_audit.py",
            "tests/public/test_real_traffic_session_audit.py",
            "experiments/real_traffic/native_comparison.py",
        ),
    ),
    (
        "Measure several real traffic weeks",
        (
            "experiments/real_traffic/series.py",
            "experiments/real_traffic/fetch_series.py",
            "experiments/real_traffic/prepare_series.py",
            "tests/public/test_real_traffic_series.py",
            ".github/workflows/real-traffic.yml",
        ),
    ),
)
EXPECTED = {path for _message, paths in GROUPS for path in paths}


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parts = sorted(STAGE.glob("part-*.b64"))
    if [part.name for part in parts] != [f"part-{index:03}.b64" for index in range(4)]:
        raise RuntimeError("staged archive parts differ")
    encoded = "".join(part.read_text() for part in parts)
    archive = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("staged archive digest differs")

    with tempfile.TemporaryDirectory(prefix="real-traffic-update-") as temporary:
        temp = Path(temporary)
        archive_path = temp / "update.tar.gz"
        archive_path.write_bytes(archive)
        with tarfile.open(archive_path, "r:gz") as bundle:
            names = set(bundle.getnames())
            if names != EXPECTED:
                raise RuntimeError(f"archive inventory differs: {sorted(names ^ EXPECTED)}")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise RuntimeError("unsafe archive member")
            bundle.extractall(temp / "source", filter="data")

        run("git", "config", "user.name", "Tugrap")
        run("git", "config", "user.email", "106766396+tugrapaydiner@users.noreply.github.com")
        for message, paths in GROUPS:
            for relative in paths:
                source = temp / "source" / relative
                target = ROOT / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            run("git", "add", "--", *paths)
            run("git", "commit", "-m", message)

    run("git", "rm", "-r", ".bootstrap/real-traffic")
    run("git", "rm", ".github/workflows/apply-real-traffic-update.yml")
    run("git", "commit", "-m", "Clean up the updater")
    run("git", "status", "--short")


if __name__ == "__main__":
    main()

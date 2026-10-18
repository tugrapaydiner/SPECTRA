from pathlib import Path
import subprocess

paths = subprocess.check_output(
    ["git", "grep", "-l", "dense_small_high_pin", "--", "*.py"],
    text=True,
).splitlines()
for name in paths:
    if name.startswith("scripts/"):
        continue
    path = Path(name)
    path.write_text(path.read_text().replace("dense_small_high_pin", "dense_tiny_high_pin"))

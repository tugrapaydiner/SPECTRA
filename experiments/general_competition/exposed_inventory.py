"""Select and acquire an exposed GBD inventory without consulting solver results.

Selection is a separate, immutable step before acquisition or execution.  The
bounded development sample is deliberately not the complete competition field,
an unopened holdout, or an official 5,000-second evaluation.  Failed acquisitions
are retained; they never cause another formula to take the selected one's place.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.request import Request, urlopen

SCHEMA = "spectra.exposed_gbd_selection.v1"
BASE_URL = "https://benchmark-database.de/file/"
SEED = "spectra-exposed-baseline-20261021-v1"
_HASH = re.compile(r"[0-9a-f]{32}\Z")
_EMPTY = {"", "empty", "None", "unknown"}
# Preserve the original CLI's selection order even after JSON serialization sorts
# the quota keys. Dict insertion order is not part of the selection contract.
STRATA = ("sat", "unsat", "unknown")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def select_records(records: list[dict], *, years: tuple[int, ...],
                   quotas: dict[str, int], seed: str = SEED) -> tuple[list[dict], list[dict]]:
    """Round-robin independent families in a stable hash order, with deduplication.

    Verdict metadata is used only to balance the declared SAT/UNSAT/UNKNOWN
    strata.  No runtime, winner, filename or current SPECTRA answer is consulted.
    GBD exact identities and each available isomorphism hash are deduplicated
    across the selected years.  These hashes are not a proof of no family leakage.
    """
    if not years or len(set(years)) != len(years) or any(type(y) is not int for y in years):
        raise ValueError("years must be distinct integers")
    if set(quotas) != {"sat", "unsat", "unknown"} or any(
        type(n) is not int or n < 0 for n in quotas.values()
    ) or not sum(quotas.values()):
        raise ValueError("declare nonnegative exact SAT/UNSAT/UNKNOWN quotas")
    if not isinstance(seed, str) or not seed:
        raise ValueError("a selection seed is required")
    unique = set()
    for row in records:
        if not _HASH.fullmatch(row["id"]):
            raise ValueError("invalid GBD identity")
        key = (row["year"], row["id"])
        if key in unique:
            raise ValueError("duplicate year/GBD metadata row")
        unique.add(key)
        if row["expected"] not in quotas or not row["family"]:
            raise ValueError("missing family or unknown verdict label")

    def rank(value: str) -> str:
        return hashlib.sha256((seed + "\0" + value).encode("utf-8")).hexdigest()

    selected, exclusions = [], []
    seen: dict[str, set[str]] = {"id": set(), "isohash": set(), "isohash2": set()}
    for year in years:
        for expected in STRATA:
            quota = quotas[expected]
            families: dict[str, list[dict]] = defaultdict(list)
            for row in records:
                if row["year"] == year and row["expected"] == expected:
                    families[row["family"]].append(row)
            queues = {
                name: deque(sorted(values, key=lambda row: (rank(row["id"]), row["id"])))
                for name, values in sorted(families.items())
            }
            order = sorted(queues, key=lambda name: (rank(name), name))
            admitted = 0
            while admitted < quota and any(queues.values()):
                for family in order:
                    if admitted >= quota:
                        break
                    queue = queues[family]
                    while queue:
                        row = queue.popleft()
                        duplicates = [key for key, values in seen.items()
                                      if row.get(key) not in _EMPTY
                                      and row.get(key) is not None
                                      and row[key] in values]
                        if duplicates:
                            exclusions.append({**row, "exclusion": "duplicate:" + ",".join(duplicates)})
                            continue
                        selected.append(dict(row))
                        for key in seen:
                            value = row.get(key)
                            if value is not None and value not in _EMPTY:
                                seen[key].add(value)
                        admitted += 1
                        break
            if admitted != quota:
                raise ValueError(f"insufficient distinct {year}/{expected}: {admitted}/{quota}")
            for queue in queues.values():
                exclusions.extend({**row, "exclusion": "predeclared_stratum_quota"} for row in queue)
    return selected, exclusions


def select(meta: Path, base: Path, output: Path, *, years: tuple[int, ...],
           quotas: dict[str, int], max_metadata_bytes: int, seed: str = SEED) -> dict:
    if type(max_metadata_bytes) is not int or max_metadata_bytes < 1:
        raise ValueError("metadata size bound must be positive")
    meta, base = meta.resolve(strict=True), base.resolve(strict=True)
    records, excluded = [], []
    with sqlite3.connect(meta.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("ATTACH DATABASE ? AS base", (base.as_uri() + "?mode=ro",))
        for year in years:
            rows = con.execute(
                "SELECT m.hash, m.family, m.result, m.isohash, m.isohash2, "
                "b.bytes, b.variables, b.clauses FROM track AS t "
                "JOIN features AS m ON m.hash=t.hash "
                "LEFT JOIN base.features AS b ON b.hash=m.hash WHERE t.value=? "
                "ORDER BY m.hash", (f"main_{year}",)
            ).fetchall()
            for raw in rows:
                row = {"id": raw["hash"], "family": raw["family"],
                       "expected": raw["result"], "isohash": raw["isohash"],
                       "isohash2": raw["isohash2"], "year": year,
                       "metadata_bytes": raw["bytes"], "variables": raw["variables"],
                       "clauses": raw["clauses"], "source_url": BASE_URL + raw["hash"]}
                try:
                    size = int(raw["bytes"])
                except (ValueError, TypeError):
                    size = 0
                if not 0 < size <= max_metadata_bytes:
                    excluded.append({**row, "exclusion": "predeclared_metadata_size_bound"})
                elif row["family"] in _EMPTY or row["expected"] not in quotas:
                    excluded.append({**row, "exclusion": "missing_stratum_metadata"})
                else:
                    records.append(row)
    chosen, quota_exclusions = select_records(records, years=years, quotas=quotas, seed=seed)
    record = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXPOSED_SELECTION_BEFORE_ACQUISITION", "seed": seed,
        "metadata": {"meta_sha256": sha256(meta), "base_sha256": sha256(base),
                     "meta_url": "https://benchmark-database.de/getdatabase/meta",
                     "base_url": "https://benchmark-database.de/getdatabase/base"},
        "years": list(years), "per_year_quotas": quotas,
        "max_metadata_bytes": max_metadata_bytes,
        "method": "stable SHA256 family round-robin within year/verdict; global exact/isohash/isohash2 deduplication",
        "selected_count": len(chosen), "eligible_memberships": len(records),
        "family_counts": dict(sorted(Counter(row["family"] for row in chosen).items())),
        "instances": chosen, "exclusions": excluded + quota_exclusions,
        "claim_boundary": "public, size-bounded development sample; not complete competition inventory or hidden confirmation",
        "near_duplicate_limit": "GBD isomorphism hashes only; no guarantee of absence of related families",
    }
    _write_new(output, record)
    return record


def acquire(selection: Path, directory: Path, *, max_source_bytes: int = 64 << 20,
            timeout: int = 60) -> dict:
    if type(max_source_bytes) is not int or max_source_bytes < 1 or type(timeout) is not int or timeout < 1:
        raise ValueError("positive exact acquisition limits required")
    # Bind acquisition to the exact bytes parsed here. Re-reading the path after a
    # long download could attribute these sources to a concurrently edited file.
    selection_bytes = selection.read_bytes()
    selection_digest = hashlib.sha256(selection_bytes).hexdigest()
    selected = json.loads(selection_bytes.decode("utf-8"))
    if not isinstance(selected, dict) or selected.get("schema") != SCHEMA:
        raise ValueError("invalid selection receipt")
    instances = selected.get("instances")
    if (not isinstance(instances, list) or not instances
            or type(selected.get("selected_count")) is not int
            or selected["selected_count"] != len(instances)
            or not isinstance(selected.get("claim_boundary"), str)
            or not selected["claim_boundary"]):
        raise ValueError("invalid selection receipt")
    identities = set()
    for item in instances:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not _HASH.fullmatch(item["id"])
                or item.get("source_url") != BASE_URL + item["id"]):
            raise ValueError("selection contains an invalid GBD source")
        if item["id"] in identities:
            raise ValueError("selection contains duplicate GBD identities")
        identities.add(item["id"])
        if (not isinstance(item.get("expected"), str) or item["expected"] not in STRATA
                or not isinstance(item.get("family"), str) or not item["family"]):
            raise ValueError("selection contains invalid stratum metadata")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "instances").mkdir()
    acquired, failures = [], []
    for item in instances:
        identity = item["id"]
        path = directory / "instances" / f"{identity}.cnf.xz"
        total = 0
        try:
            request = Request(item["source_url"], headers={"User-Agent": "SPECTRA exposed development campaign"})
            with urlopen(request, timeout=timeout) as response, path.open("xb") as stream:
                while chunk := response.read(1 << 20):
                    if total + len(chunk) > max_source_bytes:
                        raise ValueError("source exceeds predeclared download bound")
                    stream.write(chunk)
                    total += len(chunk)
            if total == 0:
                raise ValueError("empty source response")
            acquired.append({**item, "expected": item["expected"].upper(),
                             "path": path.relative_to(directory).as_posix(),
                             "sha256": sha256(path), "source_bytes": total})
        except Exception as error:
            failures.append({**item, "error_type": type(error).__name__, "error": str(error),
                             "retained_partial_bytes": total,
                             "partial_sha256": sha256(path) if path.exists() else None})
        print(json.dumps({"acquired": len(acquired), "failed": len(failures),
                          "selected": selected["selected_count"], "id": identity}), flush=True)
    receipt = {
        "schema": "spectra.exposed_gbd_acquisition.v1", "selection_sha256": selection_digest,
        "status": "ACQUIRED" if not failures else "ACQUISITION_FAILED",
        "selected_count": selected["selected_count"], "acquired_count": len(acquired),
        "max_source_bytes": max_source_bytes, "timeout_seconds": timeout,
        "failures": failures, "replacement_instances": 0,
    }
    _write_new(directory / "acquisition.json", receipt)
    # A partial download cannot silently become a smaller benchmark inventory.
    if not failures:
        _write_new(directory / "inventory.json", {
            "schema": "spectra.general_inventory.v1", "instances": acquired,
            "selection_sha256": selection_digest, "claim_boundary": selected["claim_boundary"],
        })
    else:
        _write_new(directory / "acquired-partial.json", {"instances": acquired})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--meta", type=Path, required=True)
    choose.add_argument("--base", type=Path, required=True)
    choose.add_argument("--out", type=Path, required=True)
    choose.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    choose.add_argument("--sat", type=int, default=13)
    choose.add_argument("--unsat", type=int, default=13)
    choose.add_argument("--unknown", type=int, default=6)
    choose.add_argument("--max-metadata-bytes", type=int, default=10_000_000)
    choose.add_argument("--seed", default=SEED)
    download = sub.add_parser("acquire")
    download.add_argument("--selection", type=Path, required=True)
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--max-source-bytes", type=int, default=64 << 20)
    download.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.mode == "select":
        record = select(args.meta, args.base, args.out, years=tuple(args.years),
                        quotas={"sat": args.sat, "unsat": args.unsat, "unknown": args.unknown},
                        max_metadata_bytes=args.max_metadata_bytes, seed=args.seed)
        print(json.dumps({"selected": record["selected_count"], "families": len(record["family_counts"]),
                          "selection_sha256": sha256(args.out)}, sort_keys=True))
    else:
        record = acquire(args.selection, args.out, max_source_bytes=args.max_source_bytes, timeout=args.timeout)
        print(json.dumps(record, sort_keys=True))
        if record["status"] != "ACQUIRED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

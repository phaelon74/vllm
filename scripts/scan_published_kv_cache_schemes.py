# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which published results were measured through a quantized KV cache.

Scoring used to leave ``kv_cache_dtype`` at "auto", under which vLLM resolves the
KV cache dtype from a scheme declared in the candidate's own config. A candidate
declaring one was measured with a quantized cache while every candidate it was
ranked against was not, so its KLD carries a difference in how the measurement
was taken rather than a property of the checkpoint. Laws 14 pins an unquantized
cache; this reports which already-published results predate the pin and, of
those, which ones a rescore will actually move.

    python3 scripts/scan_published_kv_cache_schemes.py --library /mnt/kld/library

Reads each report's own ``candidate_hf_repo`` and ``candidate_revision`` and
fetches only ``config.json`` and the ``hf_quant_config.json`` sidecar, so it
answers for a candidate whose weights were deleted after scoring under a lease.

Detection is ``qdq.py``'s own, not a second implementation. Vendors declare the
scheme in any of four places, and a scanner that disagreed with the inspector
would be worse than no scanner: it would clear a candidate the pipeline flags, or
flag one it clears.
"""

import argparse
import glob
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fidelity"))
# Private, deliberately: the point is to ask exactly what the inspector asks.
from qdq import _declared_kv_cache_scheme, quant_sections  # noqa: E402

CURRENT = "current"
RESCORE = "rescore"
RELABEL = "relabel"
UNKNOWN = "unknown"

CONFIG_FILES = ("config.json", "hf_quant_config.json")


def declared_kv_scheme(repo: str, revision: str | None) -> dict | None:
    """The KV cache scheme a released checkpoint declares, from its configs alone.

    Raises when the configs cannot be read. A candidate whose declaration is
    unknown must not be reported as declaring nothing, which is the reading that
    would leave a quantized-cache result published as though it were clean.
    """
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix="kv-scheme-") as staging:
        for name in CONFIG_FILES:
            try:
                path = hf_hub_download(repo, name, revision=revision)
            except Exception:
                # Only config.json is mandatory; the sidecar is vendor-specific
                # and most checkpoints have none.
                if name == CONFIG_FILES[0]:
                    raise
                continue
            with open(path, encoding="utf-8") as handle:
                payload = handle.read()
            with open(
                os.path.join(staging, name), "w", encoding="utf-8"
            ) as handle:
                handle.write(payload)
        with open(
            os.path.join(staging, CONFIG_FILES[0]), encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        return _declared_kv_cache_scheme(quant_sections(staging, config))


def classify(report: dict) -> tuple[str, str]:
    """One published report's standing against the KV cache pin.

    A report that records ``kv_cache_dtype`` at all was scored under the pin:
    ``assert_unquantized_kv_cache`` reads what the engine resolved and refuses
    both a quantized value and "auto", so the field cannot be present on a run
    that cached quantized. Absence is what dates a report, not its value.
    """
    if report.get("kv_cache_dtype"):
        return CURRENT, f"scored with kv_cache_dtype={report['kv_cache_dtype']}"

    repo = report.get("candidate_hf_repo")
    if not repo:
        return UNKNOWN, "predates the pin and records no Hub repo to ask"
    revision = report.get("candidate_revision")
    try:
        scheme = declared_kv_scheme(repo, revision)
    except Exception as exc:  # noqa: BLE001 - a read failure is not a clearance
        return UNKNOWN, f"predates the pin and {repo} could not be read ({exc})"

    if scheme is None:
        return RELABEL, "predates the pin but declares no KV scheme"
    described = scheme.get("quant_algo") or ", ".join(
        f"{key}={value}" for key, value in sorted(scheme.items()) if value
    )
    return RESCORE, f"declares {described}, so it was cached quantized"


def scan(library: str, family: str | None) -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = {
        CURRENT: [],
        RESCORE: [],
        RELABEL: [],
        UNKNOWN: [],
    }
    pattern = os.path.join(library, family or "*", "*", "report.json")
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        label = f"{parts[-3]}/{parts[-2]}"
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            found[UNKNOWN].append((label, f"report unreadable ({exc})"))
            continue
        state, detail = classify(report)
        found[state].append((label, detail))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", required=True, help="published artifact library root"
    )
    parser.add_argument(
        "--family", help="one model family; default is every family published"
    )
    args = parser.parse_args()

    found = scan(args.library, args.family)
    headings = {
        RESCORE: "RESCORE  the number is affected; its KLD includes a quantized cache",
        UNKNOWN: "UNKNOWN  cannot be cleared from here; treat as affected",
        RELABEL: "RELABEL  rescore for laws 14, but the number will reproduce",
        CURRENT: "CURRENT  already scored under the pin",
    }
    for state in (RESCORE, UNKNOWN, RELABEL, CURRENT):
        rows = found[state]
        if not rows:
            continue
        print(f"\n{headings[state]}")
        for label, detail in rows:
            print(f"  {label:<52} {detail}")

    total = sum(len(rows) for rows in found.values())
    affected = len(found[RESCORE]) + len(found[UNKNOWN])
    print(
        f"\n{total} published result(s): {len(found[CURRENT])} current, "
        f"{len(found[RELABEL])} relabel-only, {len(found[RESCORE])} affected, "
        f"{len(found[UNKNOWN])} unknown"
    )
    if not affected:
        print("no published number was measured through a quantized KV cache")
    return 1 if affected else 0


if __name__ == "__main__":
    sys.exit(main())

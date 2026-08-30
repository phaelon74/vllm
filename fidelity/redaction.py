#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One secret policy for everything that captures, renders, or publishes.

Provenance capture is deliberately broad: an artifact records the environment a
number was produced in, because a variable nobody thought mattered can move a
bitwise result. Breadth and secrecy collide directly - ``HF_TOKEN`` sits in the
same namespace as ``HF_HOME`` - so the policy is that a value whose name looks
like a credential is never recorded, never rendered, and never uploaded, while
the fact that it was set still is.

Keeping the rule in one module means capture, rendering, and publication cannot
drift apart and leave a gap between them. The module is not named ``secrets`` so
it cannot shadow the standard library module of that name.
"""

import json
import os
import re
from typing import Any

REDACTED = "<redacted>"

# Substring match on the upper-cased variable name. Deliberately broad: a false
# positive costs one line of provenance, a false negative publishes a credential.
SECRET_NAME_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "COOKIE",
    "SESSION",
    "AUTH",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "SSH_KEY",
    "_KEY",
)

# Recognizable credential shapes, for scanning content whose values we cannot
# match by name. Hugging Face tokens are first because this program pushes there.
SECRET_VALUE_PATTERNS = (
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

_TEXT_SUFFIXES = (".json", ".md", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".log")


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_NAME_MARKERS)


def redact_env(env: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Replace credential-looking values, keeping their names.

    Returns the sanitized mapping and the sorted names that were redacted, so an
    artifact can state that redaction happened rather than silently omit.
    """
    clean: dict[str, Any] = {}
    hidden: list[str] = []
    for name, value in env.items():
        if is_secret_name(name):
            clean[name] = REDACTED
            hidden.append(name)
        else:
            clean[name] = value
    return clean, sorted(hidden)


def find_secret_values(text: str) -> list[str]:
    """Names of the credential shapes present in ``text``, without echoing them."""
    return sorted({label for label, rx in SECRET_VALUE_PATTERNS if rx.search(text)})


def find_secret_fields(data: Any, path: str = "") -> list[str]:
    """Paths of credential-named JSON fields still carrying a value.

    Shape matching only catches credentials that look like the issuers we know.
    A field named like a secret holding anything other than the redaction marker
    is reported whatever its value looks like.
    """
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            here = f"{path}.{key}" if path else str(key)
            named = isinstance(key, str) and is_secret_name(key)
            if named and isinstance(value, str) and value and value != REDACTED:
                found.append(here)
            found += find_secret_fields(value, here)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found += find_secret_fields(value, f"{path}[{index}]")
    return found


def scan_tree(root: str, max_bytes: int = 8 << 20) -> list[tuple[str, str]]:
    """Every ``(relative path, credential kind)`` found in a tree's text files.

    Binary tensors are skipped: they are model weights, and reading tens of
    gigabytes to look for a token shape would make publication unusable.
    """
    findings: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith(_TEXT_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > max_bytes:
                    continue
                with open(full, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            findings += [(rel, kind) for kind in find_secret_values(text)]
            if name.lower().endswith(".json"):
                try:
                    data = json.loads(text)
                except ValueError:
                    continue
                findings += [
                    (rel, f"credential-named field {field}")
                    for field in find_secret_fields(data)
                ]
    return findings

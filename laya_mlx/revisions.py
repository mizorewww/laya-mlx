"""Supply-chain integrity for published checkpoints: opt-in pins and digest checks.

Runtime loaders keep the Hub default revision unless the caller supplies one. This preserves
compatibility with existing offline caches, including ``HF_HUB_OFFLINE=1`` deployments. The
reviewed commit SHAs below are available for callers that opt in, and every loader accepts an
optional SHA-256 map to verify artifact integrity before weights reach the runtime.

Both halves are reachable the same way. `expected_sha256` falls back to
``LAYA_SHA256_DIGESTS``, and `resolve_revision` falls back to ``LAYA_REVISION``: a commit, branch
or tag applied to every checkpoint load, or the word ``reviewed`` to use `PINNED_REVISIONS` for
whichever repository is being loaded. A `revision` argument still outranks the environment, and an
unset or empty ``LAYA_REVISION`` leaves today's behaviour exactly as it was, so a container pins
its checkpoints with one environment line and no code.

The env fallback is for checkpoints, i.e. the loads in this module's callers (`Agent`,
`ONNXAgent`). `common.build_model`'s training-time base-encoder load is deliberately excluded:
that is a different repository, and it has no reviewed SHA here to resolve to.
"""

import hashlib
import json
import ntpath
import os
from typing import Dict, Optional

# Opt-in reviewed commit SHAs of the published checkpoints. They are not applied
# implicitly, so existing Hub/offline caches keep working; pass one explicitly to pin a load.
PINNED_REVISIONS: Dict[str, str] = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}

#: Opt-in `LAYA_REVISION` value meaning "the reviewed SHA of the repository being loaded".
#: The alternative is a commit/branch/tag, which is applied to every checkpoint load instead.
REVIEWED = "reviewed"


def _reviewed_pin(model_id_or_path: str) -> Optional[str]:
    """The reviewed SHA for `model_id_or_path`, matching repo ids case-insensitively.

    Hugging Face repo ids are case-insensitive -- ``ConvaiInnovations/Laya`` resolves to the
    same repository as ``convaiinnovations/laya`` -- so a load that spells a released
    checkpoint differently must still find its pin. Refusing it instead would report a
    missing reviewed SHA for a repository that has one, which is the opposite of what the
    control exists for. The table's keys keep the canonical lowercase spelling.
    """
    pinned = PINNED_REVISIONS.get(model_id_or_path)
    if pinned is not None:
        return pinned
    wanted = model_id_or_path.lower()
    for repo, sha in PINNED_REVISIONS.items():
        if repo.lower() == wanted:
            return sha
    return None


def resolve_revision(model_id_or_path: str, revision: Optional[str] = None) -> Optional[str]:
    """Pick the revision to download.

    Both the argument and ``LAYA_REVISION`` are stripped, and the argument wins when it is truthy.
    That has one sharp edge, pinned by `tests/test_revision_pinning.py`: a truthy-but-blank
    `revision` ("   ") wins the `or` and then strips away, so it suppresses the environment and
    returns None -- a blank argument is "no pin from me, and no pin from the deployment either",
    not a way to ask for ``LAYA_REVISION``. An empty or unset one means "not asked for", so the
    variable applies. The variable holds either a commit SHA/branch/tag, applied to
    every checkpoint load the way `Router(revision=)` applies one, or the word ``reviewed``, which
    looks `model_id_or_path` up in `PINNED_REVISIONS` -- case-insensitively, the way the Hub
    itself reads repo ids, so `laya.load("ConvaiInnovations/Laya")` finds its pin.

    ``reviewed`` for a repository the table has no entry for raises rather than loading it
    unpinned: a pin that quietly resolves to nothing is the failure mode this control exists to
    prevent, and `verify_digests` refuses on the same principle. The refusal names whichever of the
    two asked for it, because telling a caller who passed `revision='reviewed'` to unset a variable
    they never set sends them to the wrong file. Everything else returns None so huggingface_hub
    applies its normal default, preserving existing online/offline caches.
    """
    value = (revision or os.environ.get("LAYA_REVISION", "")).strip()
    if not value:
        return None
    if value == REVIEWED:
        pinned = _reviewed_pin(model_id_or_path)
        if pinned is None:
            if revision:
                raise ValueError(
                    "laya: revision= asked for 'reviewed', but %r has no reviewed SHA in "
                    "PINNED_REVISIONS; pass a commit, branch or tag in revision= instead"
                    % model_id_or_path
                )
            raise ValueError(
                "laya: LAYA_REVISION=reviewed, but %r has no reviewed SHA in "
                "PINNED_REVISIONS; set LAYA_REVISION to a commit or unset it" % model_id_or_path
            )
        return pinned
    return value


def snapshot_revision(path: str) -> Optional[str]:
    """Commit SHA a Hub snapshot directory points at, or None for a plain directory.

    `snapshot_download` returns ``<cache>/snapshots/<sha>``; resolving symlinks keeps this
    correct when the snapshot entry is a link into the blob store.
    """
    real = os.path.realpath(path).rstrip(os.sep)
    parent, base = os.path.split(real)
    if os.path.basename(parent) == "snapshots" and base:
        return base
    return None


def verify_digests(
    model_dir: str,
    expected: Optional[Dict[str, str]] = None,
    onnx_path: Optional[str] = None,
) -> None:
    """Verify SHA-256 digests of files under `model_dir` against {relpath: hexdigest}.

    Raises FileNotFoundError when a listed file is absent and ValueError on a digest
    mismatch, an unsafe (absolute or escaping) relative path, or a `LAYA_SHA256_DIGESTS`
    the JSON reader cannot walk. Verification runs before any weight is parsed or
    executed, so a tampered artifact never reaches the runtime.
    """
    if expected is None:
        raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
        if not raw:
            return
        try:
            expected = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            # `RecursionError` is a third way this variable is not a document: `json` reports a
            # nesting depth it will not walk as a stack overflow, which is neither a `ValueError`
            # decode failure nor anything the caller catches. `laya/_eval_policy.py`,
            # `laya/evidence.py` and `laya/cli.py` read an unreadable JSON the same way, and here
            # the leak escaped `load()` and `Router.load()` -- which both document the ValueError
            # and mean it as "this pinning configuration is wrong" -- as an interpreter traceback.
            raise ValueError(
                "LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256"
            ) from exc
    if not isinstance(expected, dict):
        raise ValueError("expected_sha256 must be a mapping of artifact paths to digests")
    for rel, want in expected.items():
        raw_rel = str(rel).replace("\\", "/")
        if rel in ("onnx", "onnx_path") and onnx_path:
            path = onnx_path
        else:
            if raw_rel.startswith("/") or os.path.isabs(raw_rel) or ntpath.isabs(raw_rel):
                raise ValueError("laya: unsafe absolute path in expected digests: %r" % (rel,))
            rel_norm = raw_rel.lstrip("/")
            if not rel_norm or rel_norm == ".." or rel_norm.startswith("../") or "/../" in rel_norm:
                raise ValueError("laya: unsafe path in expected digests: %r" % (rel,))
            # Drive-anchored components are not absolute by ntpath's own test
            # (ntpath.isabs("C:evil") is False), but on Windows ntpath.join resets to
            # the drive the moment one appears -- os.path.join("C:\\dir", "a", "b:c")
            # is "b:c" -- so a digest entry like "a/b:c" escapes model_dir and the
            # verification reads a file outside the checkpoint it is meant to gate.
            if any(ntpath.splitdrive(part)[0] for part in rel_norm.split("/")):
                raise ValueError(
                    "laya: unsafe drive-anchored path in expected digests: %r" % (rel,)
                )
            path = os.path.join(model_dir, rel_norm)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "laya: cannot verify %r: no such file under %s" % (rel, model_dir)
            )
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        got = digest.hexdigest()
        if got.lower() != str(want).strip().lower():
            raise ValueError(
                "laya: SHA-256 mismatch for %s: expected %s, got %s. The artifact does "
                "not match the reviewed digest; refusing to load it." % (rel, want, got)
            )

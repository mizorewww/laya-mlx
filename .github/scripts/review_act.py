"""Carry out the automated reviewer's decision as the maintainer.

Claude only reads the pull request and returns a decision (see pr-review.yml).
This script posts it with the maintainer's token: one review with inline
comments, approved when Claude says LGTM, or a comment and close, and it commits
and pushes the edits Claude proposed. An approved pull request is merged once CI
passes (merge_ready.py). The decision is untrusted input, so edits are applied
as exact string replacements or deletions inside the PR checkout only.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from merge_ready import merge_if_ready

REPO = os.environ["GITHUB_REPOSITORY"]
PR = os.environ["PR"]
TOKEN = os.environ["MAINTAINER_TOKEN"]
ACT = os.environ["ACT"] == "true"
FIX = os.environ["FIX"] == "true"
HEAD_REPO = os.environ["HEAD_REPO"]
HEAD_REF = os.environ["HEAD_REF"]
HEAD_SHA = os.environ["HEAD_SHA"]
AUTHOR = os.environ["AUTHOR"]
CO_AUTHOR = os.environ["CO_AUTHOR"]
WORK = (Path(os.environ["GITHUB_WORKSPACE"]) / "pr").resolve()
GIT_ENV = {**os.environ, "GIT_DIR": os.environ["PR_GIT_DIR"], "GIT_WORK_TREE": str(WORK)}
FOOTER = "\n\n<sub>Automated review drafted by Claude Code.</sub>"


def api(method, path, body=None):
    request = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read() or "null")


def git(*args):
    return subprocess.run(
        ["git", *args], env=GIT_ENV, check=True, capture_output=True, text=True
    ).stdout


def planned_edits(edits):
    """Return {path: new text, or None to delete} for all edits, or raise ValueError.

    Nothing is written.
    """
    files = {}
    for edit in edits:
        path = (WORK / edit["path"]).resolve()
        if WORK not in path.parents or ".git" in path.relative_to(WORK).parts:
            raise ValueError(f"{edit['path']}: outside the pull request checkout")
        if path not in files:
            files[path] = path.read_text() if path.exists() else None
        text, old = files[path], edit.get("old", "")
        if edit.get("delete"):
            if text is None:
                raise ValueError(f"{edit['path']}: cannot delete a missing file")
            files[path] = None
        elif old == "":
            if text is not None:
                raise ValueError(f"{edit['path']}: empty `old` but the file exists")
            files[path] = edit.get("new", "")
        elif text is None or text.count(old) != 1:
            raise ValueError(f"{edit['path']}: `old` does not occur exactly once")
        else:
            files[path] = text.replace(old, edit.get("new", ""), 1)
    return files


def push_fix(out):
    """Commit and push Claude's edits. Return (new head sha or None, note for the review)."""
    try:
        files = planned_edits(out.get("edits") or [])
    except (ValueError, OSError) as error:
        return None, f"\n\nThe proposed fix was not applied: {error}."
    for path, text in files.items():
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    git("add", "-A")
    message = out.get("commit_message") or "Apply automated review fixes"
    git(
        "-c",
        "user.name=claude[bot]",
        "-c",
        "user.email=41898282+claude[bot]@users.noreply.github.com",
        "commit",
        "-q",
        "--author",
        AUTHOR,
        "--trailer",
        CO_AUTHOR,
        "-m",
        message,
    )
    patch = git("diff", "HEAD~1")
    url = f"https://x-access-token:{TOKEN}@github.com/{HEAD_REPO}.git"
    if subprocess.run(
        ["git", "push", "-q", url, f"HEAD:refs/heads/{HEAD_REF}"], env=GIT_ENV, capture_output=True
    ).returncode:
        return None, (
            f"\n\nI could not push the fix to `{HEAD_REPO}:{HEAD_REF}`; "
            f"here it is as a patch:\n\n```diff\n{patch[:60000]}\n```"
        )
    return git("rev-parse", "HEAD").strip(), "\n\nI pushed these fixes to the branch."


def as_list(findings):
    if not findings:
        return ""
    return "\n" + "".join(f"\n- `{f['path']}:{f['line']}`: {f['body']}" for f in findings)


def main():
    out = json.loads(os.environ.get("OUTPUT") or "{}")
    if not out:
        print("No decision from the reviewer.")
        return
    if api("GET", f"repos/{REPO}/pulls/{PR}")["state"] != "open":
        print("Pull request is not open.")
        return

    action = out.get("action", "review")
    if action == "close" and not ACT or action == "fix" and not FIX:
        action = "review"
    body = out["summary"]

    if action == "close":
        api("POST", f"repos/{REPO}/issues/{PR}/comments", {"body": body + FOOTER})
        api("PATCH", f"repos/{REPO}/pulls/{PR}", {"state": "closed"})
        print("Closed.")
        return

    commit_id, findings = HEAD_SHA, out.get("findings") or []
    fixed = True
    if action == "fix":
        new_sha, note = push_fix(out)
        body += note
        fixed = new_sha is not None
        if fixed:
            # Inline comments point at the old head; list them in the body instead.
            commit_id, body, findings = new_sha, body + as_list(findings), []

    approve = ACT and out.get("lgtm") is True and fixed
    if approve:
        if not body.lstrip().upper().startswith("LGTM"):
            body = "LGTM\n\n" + body
        body += "\n\nThis will be merged automatically once CI passes."
    review = {
        "commit_id": commit_id,
        "body": body + FOOTER,
        "event": "APPROVE" if approve else "COMMENT",
        "comments": [
            {"path": f["path"], "line": f["line"], "side": "RIGHT", "body": f["body"]}
            for f in findings
        ],
    }
    try:
        api("POST", f"repos/{REPO}/pulls/{PR}/reviews", review)
    except urllib.error.HTTPError as error:
        if error.code != 422 or not findings:
            raise
        # A line outside the diff rejects the whole review; fall back to a list.
        review["body"] = body + as_list(findings) + FOOTER
        review["comments"] = []
        api("POST", f"repos/{REPO}/pulls/{PR}/reviews", review)
    print(f"Posted review: {review['event']}.")
    if approve and commit_id == HEAD_SHA:
        # Nothing was pushed, so CI may already have passed on this head.
        merge_if_ready(PR)


if __name__ == "__main__":
    main()

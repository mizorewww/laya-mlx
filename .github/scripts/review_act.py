"""Carry out the automated reviewer's decision as the maintainer.

Claude only reads the pull request and returns a decision (see pr-review.yml).
This script acts on it with the maintainer's token:

- close: comment and close.
- adopt: take the change without waiting for the author. pr/ holds the base
  branch with the pull request merged in (conflicts left as markers); Claude's
  edits resolve conflicts, remove redundancy and fix problems. The result is
  committed as the maintainer with Claude as co-author and pushed to the pull
  request's branch when the author allows edits, otherwise to adopt/pr-<n> in
  this repository with a new pull request. merge_ready.py merges it once CI passes.
- review: one review with inline comments; the maintainer decides.

The decision is untrusted input, so edits are applied as exact string
replacements or deletions inside the checkout only.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from merge_ready import ADOPT_PREFIX, MARKER, merge_if_ready, touches_github

REPO = os.environ["GITHUB_REPOSITORY"]
PR = os.environ["PR"]
TOKEN = os.environ["MAINTAINER_TOKEN"]
ACT = os.environ["ACT"] == "true"
WRITABLE = os.environ["WRITABLE"] == "true"
HEAD_REPO = os.environ["HEAD_REPO"]
HEAD_REF = os.environ["HEAD_REF"]
HEAD_SHA = os.environ["HEAD_SHA"]
CONFLICTS = [path for path in os.environ.get("CONFLICTS", "").split("\n") if path]
AUTHOR = os.environ["AUTHOR"]
CO_AUTHOR = os.environ["CO_AUTHOR"]
WORK = (Path(os.environ["GITHUB_WORKSPACE"]) / "pr").resolve()
GIT_ENV = {**os.environ, "GIT_DIR": os.environ["PR_GIT_DIR"], "GIT_WORK_TREE": str(WORK)}
FOOTER = f"\n\n<sub>{MARKER}</sub>"


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


def git(*args, check=True):
    return subprocess.run(["git", *args], env=GIT_ENV, check=check, capture_output=True, text=True)


def planned_edits(edits):
    """Return {path: new text, or None to delete} for all edits, or raise ValueError.

    Nothing is written.
    """
    files = {}
    for number, edit in enumerate(edits, 1):
        where = f"edit {number} ({edit['path']})"
        path = (WORK / edit["path"]).resolve()
        if WORK not in path.parents or ".git" in path.relative_to(WORK).parts:
            raise ValueError(f"{where}: outside the pull request checkout")
        if path not in files:
            files[path] = path.read_text() if path.exists() else None
        text, old = files[path], edit.get("old", "")
        if edit.get("delete"):
            if text is None:
                raise ValueError(f"{where}: cannot delete a missing file")
            files[path] = None
        elif "content" in edit:
            files[path] = edit["content"]
        elif old == "":
            if text is not None:
                raise ValueError(f"{where}: empty `old` but the file exists")
            files[path] = edit.get("new", "")
        elif text is None or text.count(old) != 1:
            found = 0 if text is None else text.count(old)
            raise ValueError(f"{where}: `old` occurs {found} times, not once")
        else:
            files[path] = text.replace(old, edit.get("new", ""), 1)
    return files


def unresolved():
    """Conflicted files that still hold conflict markers."""
    left = []
    for name in CONFLICTS:
        path = WORK / name
        if path.exists() and any(
            line.startswith(("<<<<<<< ", ">>>>>>> ")) for line in path.read_text().splitlines()
        ):
            left.append(name)
    return left


def push(url, ref):
    result = subprocess.run(
        ["git", "push", "-q", "--force", url, f"HEAD:refs/heads/{ref}"],
        env=GIT_ENV,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(f"Push to {ref} failed: {result.stderr.replace(TOKEN, '***').strip()}")
    return not result.returncode


def approve(sha, body):
    api(
        "POST",
        f"repos/{REPO}/pulls/{PR}/reviews",
        {"commit_id": sha, "event": "APPROVE", "body": "LGTM\n\n" + body + FOOTER},
    )


def adopt(out, pr):
    """Take the change over. Return what happened, or raise ValueError."""
    if touches_github(PR):
        raise ValueError("it changes files under .github/, which is never merged automatically")
    edits = out.get("edits") or []
    if not edits and not CONFLICTS:
        # The pull request merges as it is.
        git("merge", "--abort", check=False)
        approve(HEAD_SHA, out["summary"])
        merge_if_ready(PR)
        return "approved as it is"

    for path, text in planned_edits(edits).items():
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if left := unresolved():
        raise ValueError(f"conflicts remain in {', '.join(left)}")
    git("add", "-A")
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
        out.get("commit_message") or f"Adopt #{PR}",
    )
    sha = git("rev-parse", "HEAD").stdout.strip()

    if WRITABLE and push(f"https://x-access-token:{TOKEN}@github.com/{HEAD_REPO}.git", HEAD_REF):
        approve(sha, out["summary"] + "\n\nI pushed these changes to your branch.")
        return "pushed to the pull request"

    branch = f"{ADOPT_PREFIX}{PR}"
    if not push(f"https://x-access-token:{TOKEN}@github.com/{REPO}.git", branch):
        raise ValueError(f"could not push {branch}")
    owner = REPO.split("/")[0]
    existing = api("GET", f"repos/{REPO}/pulls?state=open&head={owner}:{branch}")
    number = (
        existing[0]["number"]
        if existing
        else api(
            "POST",
            f"repos/{REPO}/pulls",
            {
                "title": pr["title"],
                "head": branch,
                "base": pr["base"]["ref"],
                "body": f"Adopts #{PR} by @{pr['user']['login']}. It merges once CI passes, "
                f"and #{PR} is closed then.\n\n" + out["summary"] + FOOTER,
            },
        )["number"]
    )
    api(
        "POST",
        f"repos/{REPO}/issues/{PR}/comments",
        {
            "body": out["summary"] + "\n\nI could not push to this branch, so I continued it "
            f"in #{number}. It merges once CI passes, and this pull request is closed then. "
            "Thank you!" + FOOTER
        },
    )
    return f"continued in #{number}"


def as_list(findings):
    if not findings:
        return ""
    return "\n" + "".join(f"\n- `{f['path']}:{f['line']}`: {f['body']}" for f in findings)


def review(out, note=""):
    findings = out.get("findings") or []
    body = {
        "commit_id": HEAD_SHA,
        "body": out["summary"] + note + FOOTER,
        "event": "COMMENT",
        "comments": [
            {"path": f["path"], "line": f["line"], "side": "RIGHT", "body": f["body"]}
            for f in findings
        ],
    }
    try:
        api("POST", f"repos/{REPO}/pulls/{PR}/reviews", body)
    except urllib.error.HTTPError as error:
        if error.code != 422 or not findings:
            raise
        # A line outside the diff rejects the whole review; fall back to a list.
        body["body"] = out["summary"] + note + as_list(findings) + FOOTER
        body["comments"] = []
        api("POST", f"repos/{REPO}/pulls/{PR}/reviews", body)


def main():
    out = json.loads(os.environ.get("OUTPUT") or "{}")
    if not out:
        print("No decision from the reviewer.")
        return
    pr = api("GET", f"repos/{REPO}/pulls/{PR}")
    if pr["state"] != "open":
        print("Pull request is not open.")
        return
    action = out.get("action", "review")
    if action != "review" and not ACT:
        action = "review"

    if action == "close":
        api("POST", f"repos/{REPO}/issues/{PR}/comments", {"body": out["summary"] + FOOTER})
        api("PATCH", f"repos/{REPO}/pulls/{PR}", {"state": "closed"})
        print("Closed.")
    elif action == "adopt":
        try:
            print(f"Adopted: {adopt(out, pr)}.")
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            review(out, f"\n\nI could not take this over automatically: {error}.")
            print(f"Not adopted: {error}")
    else:
        review(out)
        print("Posted review.")


if __name__ == "__main__":
    main()

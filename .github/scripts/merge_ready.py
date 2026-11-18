"""Merge a contributor's pull request that the automated review took over, once CI passes.

Called by review_act.py right after an approval, and by pr-merge.yml whenever a
CI run finishes. It merges only when all of these hold for the pull request's
current head:

- it is open, changes nothing under .github/, and either comes from a contributor
  outside the repository with the automation's approval of that exact commit as
  the maintainer's latest review, or is an adopt/pr-<n> pull request the
  automation opened (then the adopted pull request is closed after the merge);
- the latest run of each CI workflow (CI_WORKFLOWS, comma-separated) on that commit
  succeeded, and at least one ran. With no CI workflows configured, the approval
  alone is enough.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["MAINTAINER_TOKEN"]
CI_WORKFLOWS = {
    name.strip() for name in os.environ.get("CI_WORKFLOWS", "").split(",") if name.strip()
}
MARKER = "Automated review drafted by Claude Code."
TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}
ADOPT_PREFIX = "adopt/pr-"


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


def touches_github(number):
    files = api("GET", f"repos/{REPO}/pulls/{number}/files?per_page=100")
    return len(files) == 100 or any(
        f["filename"].startswith(".github/")
        or f.get("previous_filename", "").startswith(".github/")
        for f in files
    )


def adopted(pr, me):
    """The pull request an adopt/pr-<n> pull request continues, or None."""
    match = re.match(r"Adopts #(\d+) ", pr["body"] or "")
    if (
        pr["head"]["ref"].startswith(ADOPT_PREFIX)
        and pr["head"]["repo"]["full_name"] == REPO
        and pr["user"]["login"] == me
        and MARKER in (pr["body"] or "")
        and match
    ):
        return match.group(1)
    return None


def not_ready(pr, me):
    """Why the pull request cannot be merged yet, or None."""
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or pr["draft"]:
        return "not open"
    if touches_github(pr["number"]):
        return "it changes .github/"
    if not adopted(pr, me):
        if pr["author_association"] in TRUSTED:
            return "collaborators merge their own pull requests"
        mine = [
            r
            for r in api("GET", f"repos/{REPO}/pulls/{pr['number']}/reviews?per_page=100")
            if r["user"]["login"] == me
        ]
        if not mine or not (
            mine[-1]["state"] == "APPROVED"
            and mine[-1]["commit_id"] == sha
            and MARKER in (mine[-1]["body"] or "")
        ):
            return "no automated approval of the current head"
    if not CI_WORKFLOWS:
        return None
    # Path filters may skip some workflows; each one that ran must have passed.
    runs = api("GET", f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=100")["workflow_runs"]
    latest = {}
    for run in runs:
        if run["name"] in CI_WORKFLOWS and run["event"] == "pull_request":
            if run["name"] not in latest or run["created_at"] > latest[run["name"]]["created_at"]:
                latest[run["name"]] = run
    if not latest:
        return "CI has not run on the current head"
    for name, run in sorted(latest.items()):
        if run["conclusion"] != "success":
            return f"{name} is {run['conclusion'] or run['status']}"
    return None


def merge_if_ready(number):
    pr = api("GET", f"repos/{REPO}/pulls/{number}")
    me = api("GET", "user")["login"]
    reason = not_ready(pr, me)
    if reason:
        print(f"#{number} not merged: {reason}.")
        return False
    try:
        api(
            "PUT",
            f"repos/{REPO}/pulls/{number}/merge",
            {
                "merge_method": "squash",
                "sha": pr["head"]["sha"],
                "commit_title": f"{pr['title']} (#{number})",
            },
        )
    except urllib.error.HTTPError as error:
        # 405: not mergeable (conflicts); 409: the head moved since the check; 403: the
        # token lacks a scope (merging a branch that predates a workflow change needs
        # `workflow`).
        message = json.loads(error.read() or "{}").get("message", "")
        print(f"#{number} not merged: GitHub refused the merge ({error.code}): {message}")
        return False
    print(f"#{number} merged.")
    if original := adopted(pr, me):
        api(
            "POST",
            f"repos/{REPO}/issues/{original}/comments",
            {"body": f"Merged in #{number}. Thank you!\n\n<sub>{MARKER}</sub>"},
        )
        api("PATCH", f"repos/{REPO}/pulls/{original}", {"state": "closed"})
    return True


def pulls_for(sha):
    pulls = api("GET", f"repos/{REPO}/pulls?state=open&per_page=100")
    return [p["number"] for p in pulls if p["head"]["sha"] == sha]


if __name__ == "__main__":
    for number in pulls_for(sys.argv[1]):
        merge_if_ready(number)

"""Merge a contributor's pull request that the automated review approved, once CI passes.

Called by review_act.py right after an approval, and by pr-merge.yml whenever a
CI run finishes. It merges only when all of these hold for the pull request's
current head: the pull request is open and comes from a contributor outside the
repository, the maintainer's latest review is the automation's approval of that
exact commit, and the latest CI run on that commit succeeded.
"""

import json
import os
import sys
import urllib.error
import urllib.request

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["MAINTAINER_TOKEN"]
CI_WORKFLOW = os.environ["CI_WORKFLOW"]
MARKER = "Automated review drafted by Claude Code."
TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}


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


def not_ready(pr):
    """Why the pull request cannot be merged yet, or None."""
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or pr["draft"]:
        return "not open"
    if pr["author_association"] in TRUSTED:
        return "collaborators merge their own pull requests"
    me = api("GET", "user")["login"]
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
    runs = api("GET", f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=100")["workflow_runs"]
    runs = [r for r in runs if r["name"] == CI_WORKFLOW and r["event"] == "pull_request"]
    if not runs:
        return "CI has not run on the current head"
    latest = max(runs, key=lambda r: r["created_at"])
    if latest["conclusion"] != "success":
        return f"CI is {latest['conclusion'] or latest['status']}"
    return None


def merge_if_ready(number):
    pr = api("GET", f"repos/{REPO}/pulls/{number}")
    reason = not_ready(pr)
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
    return True


def pulls_for(sha):
    pulls = api("GET", f"repos/{REPO}/pulls?state=open&per_page=100")
    return [p["number"] for p in pulls if p["head"]["sha"] == sha]


if __name__ == "__main__":
    for number in pulls_for(sys.argv[1]):
        merge_if_ready(number)

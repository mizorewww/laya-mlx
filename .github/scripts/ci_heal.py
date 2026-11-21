"""Decide what to do about a completed CI or port run (.github/workflows/ci-heal.yml).

Acts as the maintainer (MAINTAINER_TOKEN), so the runs it starts trigger workflows.
Prints `action=fix` to GITHUB_OUTPUT when Claude must fix main, a ci-fix/ branch, or a
pull request the maintainer approved at its current head (one the review adopted;
pr-merge.yml merges it once CI passes). Everything else it does itself:
- a run on a commit that is no longer its branch's head is ignored, and so is any other
  branch or pull request: its code is not trusted with this job's secrets;
- jobs that never got a runner, were cancelled or timed out are rerun;
- a real failure on a renovate/ branch, or a port run that did not finish, starts
  the port workflow (which fixes CI when its commit is already on the branch);
- a ci-fix/ pull request is merged once every CI workflow passed on its head.
"""

import json
import os
import sys
import urllib.error
import urllib.request

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["MAINTAINER_TOKEN"]
RUN_ID = os.environ["RUN_ID"]
CI_WORKFLOWS = {name.strip() for name in os.environ["CI_WORKFLOWS"].split(",") if name.strip()}
PORT_WORKFLOW = os.environ.get("PORT_WORKFLOW", "")  # file name, empty without ports
PORT_NAME = "Upstream port"
PORT_BRANCH_INPUT = os.environ.get("PORT_BRANCH_INPUT") == "true"
FAILED = {"failure", "cancelled", "timed_out", "startup_failure"}
MAX_ATTEMPTS = 5  # a run whose jobs keep failing for the runner goes to Claude after this
MAX_PORT_RESTARTS = 3  # per commit
MAX_FIX_COMMITS = 5  # on one ci-fix/ branch
MAX_RUN_ATTEMPTS = 8  # reruns of one run, by this script or a fix that changed nothing


def api(method, path, body=None):
    request = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read() or "null")


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        for key, value in values.items():
            out.write(f"{key}={value}\n")


def done(message):
    print(message)
    sys.exit(0)


def open_pr(branch):
    pulls = api("GET", f"repos/{REPO}/pulls?state=open&head={REPO.split('/')[0]}:{branch}")
    return pulls[0] if pulls else None


def port_busy(branch):
    # Recent runs unfiltered: the status filter misses a run dispatched seconds ago.
    runs = api("GET", f"repos/{REPO}/actions/workflows/{PORT_WORKFLOW}/runs?per_page=20")
    return any(
        r["status"] != "completed"
        and (r["head_branch"] == branch or r["event"] == "workflow_dispatch")
        for r in runs["workflow_runs"]
    )


def start_port(branch, fix_run=""):
    if port_busy(branch):
        done(f"A port run is already queued or running for {branch}.")
    inputs = {"fix_run": str(fix_run)}
    if PORT_BRANCH_INPUT:
        inputs["branch"] = branch
    api(
        "POST",
        f"repos/{REPO}/actions/workflows/{PORT_WORKFLOW}/dispatches",
        {"ref": "main", "inputs": inputs},
    )
    done(f"Started {PORT_WORKFLOW} on {branch}" + (f" to fix run {fix_run}." if fix_run else "."))


def approved_pr(sha):
    """The open pull request whose head is sha, approved there by the maintainer."""
    owner = REPO.split("/")[0]
    for pr in api("GET", f"repos/{REPO}/pulls?state=open&per_page=100"):
        if pr["head"]["sha"] != sha or pr["draft"]:
            continue
        reviews = api("GET", f"repos/{REPO}/pulls/{pr['number']}/reviews?per_page=100")
        if any(
            r["user"]["login"] == owner and r["state"] == "APPROVED" and r["commit_id"] == sha
            for r in reviews
        ):
            return pr
    return None


def ci_green(sha):
    runs = api("GET", f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=100")["workflow_runs"]
    latest = {}
    for run in sorted(runs, key=lambda r: r["created_at"]):
        if run["name"] in CI_WORKFLOWS:
            latest[run["name"]] = run
    return bool(latest) and all(r["conclusion"] == "success" for r in latest.values())


def main():
    run = api("GET", f"repos/{REPO}/actions/runs/{RUN_ID}")
    branch, sha, name, conclusion = (
        run["head_branch"],
        run["head_sha"],
        run["name"],
        run["conclusion"],
    )
    print(
        f"{name} run {RUN_ID} (attempt {run['run_attempt']}) on {branch} at {sha[:7]}: {conclusion}"
    )

    if name == PORT_NAME:
        # Its head_branch is the PR branch, or main for a dispatch: then check every open port.
        if conclusion not in FAILED or not PORT_WORKFLOW:
            done("Nothing to do for this port run.")
        if branch.startswith("renovate/"):
            branches = [branch]
        else:
            branches = [
                p["head"]["ref"]
                for p in api("GET", f"repos/{REPO}/pulls?state=open&per_page=100")
                if p["head"]["ref"].startswith("renovate/")
                and p["head"]["repo"]["full_name"] == REPO
            ]
        for port_branch in branches:
            pr = open_pr(port_branch)
            if not pr or any(label["name"] == "port-ready" for label in pr["labels"]):
                continue
            commit = pr["head"]["sha"]
            comments = api("GET", f"repos/{REPO}/issues/{pr['number']}/comments?per_page=100")
            tried = sum(f"port restart for {commit[:12]}" in c["body"] for c in comments)
            if tried >= MAX_PORT_RESTARTS:
                print(
                    f"{port_branch}: the port did not finish {tried} times at {commit[:7]}; leaving it."
                )
                continue
            if port_busy(port_branch):
                continue
            api(
                "POST",
                f"repos/{REPO}/issues/{pr['number']}/comments",
                {
                    "body": f"The port run did not finish ({run['html_url']}); starting it again.\n\n"
                    f"<sub>Automated port restart for {commit[:12]}.</sub>",
                },
            )
            start_port(port_branch)
        done("No unfinished port to restart.")

    ours = run["head_repository"]["full_name"] == REPO
    adopted = None
    if not ours or (branch != "main" and not branch.startswith(("renovate/", "ci-fix/"))):
        adopted = approved_pr(sha)
        if not adopted:
            done("Not an automation branch, nor a pull request approved at this commit.")
        branch = adopted["head"]["ref"]
    else:
        try:
            head = api("GET", f"repos/{REPO}/branches/{branch}")["commit"]["sha"]
        except urllib.error.HTTPError:
            done(f"{branch} no longer exists.")
        if head != sha:
            done(f"{branch} moved to {head[:7]}; its own CI decides.")

    if conclusion == "success":
        pr = open_pr(branch) if ours and branch.startswith("ci-fix/") else None
        if pr and ci_green(sha):
            api(
                "PUT",
                f"repos/{REPO}/pulls/{pr['number']}/merge",
                {
                    "merge_method": "squash",
                    "sha": sha,
                    "commit_title": f"{pr['title']} (#{pr['number']})",
                },
            )
            api("DELETE", f"repos/{REPO}/git/refs/heads/{branch}")
            done(f"Merged #{pr['number']}.")
        done("Nothing to do.")
    if conclusion not in FAILED:
        done(f"Nothing to do for {conclusion}.")

    if run["run_attempt"] > MAX_RUN_ATTEMPTS:
        done(f"Attempt {run['run_attempt']} failed too; leaving it to the maintainer.")
    jobs = api(
        "GET", f"repos/{REPO}/actions/runs/{RUN_ID}/attempts/{run['run_attempt']}/jobs?per_page=100"
    )["jobs"]
    failed = [j for j in jobs if j["conclusion"] in FAILED]
    # A job that failed in one of its own steps is a real failure; any other failed job
    # (cancelled, timed out, never started, runner lost) is the runner's.
    real = [j for j in failed if any(s["conclusion"] == "failure" for s in j["steps"])]
    if failed and not real and run["run_attempt"] < MAX_ATTEMPTS:
        api("POST", f"repos/{REPO}/actions/runs/{RUN_ID}/rerun-failed-jobs")
        done("Reran " + ", ".join(j["name"] for j in failed) + ": they did not fail in a step.")
    print("Failed in a step: " + ", ".join(j["name"] for j in real or failed))

    if adopted:
        comments = api("GET", f"repos/{REPO}/issues/{adopted['number']}/comments?per_page=100")
        tries = sum("Automated CI fix" in c["body"] for c in comments)
        if tries >= MAX_FIX_COMMITS:
            done(f"#{adopted['number']} had {tries} fixes; leaving it to the maintainer.")
        output(
            action="fix",
            branch=branch,
            head_repo=adopted["head"]["repo"]["full_name"],
            pr=adopted["number"],
        )
        done(f"Claude fixes #{adopted['number']}.")
    if branch.startswith("renovate/"):
        if PORT_WORKFLOW:
            start_port(branch, RUN_ID)
        done("No port workflow; Renovate's own failure waits for the maintainer.")
    if branch == "main":
        for pr in api("GET", f"repos/{REPO}/pulls?state=open&per_page=100"):
            if pr["head"]["ref"].startswith("ci-fix/") and pr["head"]["repo"]["full_name"] == REPO:
                done(f"#{pr['number']} is already fixing main.")
    else:
        commits = api("GET", f"repos/{REPO}/compare/main...{branch}")["ahead_by"]
        if commits >= MAX_FIX_COMMITS:
            done(f"{branch} has {commits} fix commits; leaving it to the maintainer.")
    output(action="fix", branch=branch)
    print("Claude fixes it.")


if __name__ == "__main__":
    main()

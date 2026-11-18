# Contributing to Laya-MLX

This project follows upstream [Laya](https://github.com/NandhaKishorM/laya) behavior through a
native MLX implementation. Upstream-compatible fixes take priority over new model variants,
service APIs and demos. If a change has to differ from upstream, say why in the pull request,
add a regression test, and record the divergence in [docs/UPSTREAM.md](docs/UPSTREAM.md).

## Development setup

```bash
gh repo clone mizorewww/laya-mlx
cd laya-mlx
uv sync --extra dev --extra reference --extra demo --extra serve --extra mcp --extra structured --extra llamaindex
source .venv/bin/activate

# Upstream reference used by parity and contract tests (the commit CI pins):
gh repo clone NandhaKishorM/laya .upstream
git -C .upstream checkout b1d656952f0f9581ea691a70cb6f38abf3eb13a9
```

Model weights are downloaded separately and are never committed.

## Tests

```bash
ruff check . && ruff format --check .
pytest -q tests/test_official_host.py tests/test_official_contract.py   # upstream host contracts
pytest -q --ignore=tests/test_official_contract.py --ignore=tests/test_official_host.py
python scripts/audit_public_api.py --package laya_mlx                    # must report 0 unclassified
```

Unit tests use small random models, including direct comparisons with Transformers and the pinned
upstream decision head. Upstream's timing-sensitive host contracts run in their own process, before
the native tests; keep every upstream assertion and timeout unchanged.
[UPSTREAM.md](docs/UPSTREAM.md#idle-window-timing) explains the known idle-window timing
instability.

On Linux, install `.[dev,serve,mcp,structured]` and run
`pytest -q tests/test_official_host.py tests/test_upstream_0417.py tests/test_upstream_0434.py`.
These use real host code with stub inference and say nothing about MLX numerical parity.

GitHub Actions runs the host checks on Linux and small-model CPU tests on macOS arm64. Full-checkpoint
GPU validation and benchmarks run locally only.

## Real checkpoints and benchmarks

```bash
uv sync --extra benchmark  # in addition to the extras above
python -m benchmarks.download                 # pinned original checkpoints into models/
python -m benchmarks.validate --repeats 100   # 63 fixtures x 3 models x FP32/FP16, plus stability
python -m benchmarks.run --iterations 50 --warmup 5
python -m benchmarks.accuracy --per-class 64
python -m benchmarks.report
```

Run GPU measurements one at a time. Each backend/checkpoint runs in a fresh process, and every timing
sample is stored in [benchmarks/results](benchmarks/results). [BENCHMARKS.md](BENCHMARKS.md) explains
the timing boundaries. Research scripts and raw measurements are in [experiments/](experiments), with
write-ups indexed in [docs/README.md](docs/README.md).

## Publishing converted checkpoints

```bash
python -m scripts.prepare_hub --account YOUR_HF_USERNAME
hf upload YOUR_HF_USERNAME/laya-mlx models/hub/laya-mlx . --exclude '.cache/*'
hf cache verify YOUR_HF_USERNAME/laya-mlx --local-dir models/hub/laya-mlx
```

The preparation script checks every exported tensor against its FP16 source. Upload the other two
prepared folders the same way.

## Releasing

1. Run the tests above and `python -m benchmarks.validate --repeats 100` on the release commit.
2. Set the same version in `pyproject.toml`, `laya_mlx/__init__.py` and `uv.lock` (`uv lock`).
3. Add a section to [CHANGELOG.md](CHANGELOG.md).
4. Push a matching tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The release workflow reruns CI, checks that the tag and all three versions agree, builds and
`twine check`s the wheel and sdist, publishes to PyPI with the `PYPI_API_TOKEN` secret, and creates a
GitHub release. Any failure stops the release before publishing.

Upstream releases are ported and released without a maintainer. Renovate, which
`.github/workflows/renovate.yml` runs hourly with the maintainer's token, opens a PR on
`renovate/upstream-laya` for each upstream tag, Claude Code ports it there following
[.github/upstream-port.md](.github/upstream-port.md), and when every check passes and CI is green the
PR is merged, tagged and released. Real-checkpoint validation and the wheel smoke test are not part
of that path; run them afterwards.

New issues get an automated first reply, and `@claude` in a comment asks Claude Code for more.
Pull requests get an automated review. Claude only reads and decides; replies, closes, approvals,
commits and merges are made under the maintainer's account with the `MAINTAINER_TOKEN` secret.
A pull request from a contributor outside the repository is closed when it is not needed, and
otherwise taken over without waiting for the author: conflicts with `main` resolved, redundant
code and tests removed, problems fixed, committed as the maintainer with Claude as co-author, and
squash-merged once CI passes. The changes go to the contributor's branch when "Allow edits by
maintainers" is on, else to an `adopt/pr-<n>` pull request that closes the original when it
merges. Pull requests that change `.github/` are never merged automatically, and collaborators'
pull requests get a review only.

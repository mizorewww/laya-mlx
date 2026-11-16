"""Audit public signatures/exports against the pinned source without importing optional stacks.

This is a surface audit, not a substitute for official assertions or numerical tests.
Backend tensor classes and implementation modules have explicit native replacements.
"""

import argparse
import ast
import copy
import json
import subprocess
from pathlib import Path

PIN = "80a1da1fa5448ea12163f4d977935625a0aea670"
BACKEND_MODULES = {
    "train.py",
    "train_cli.py",
    "fast.py",
    "_compile.py",
    "onnx_agent.py",
    "tl_kernels.py",
    # vegaml's engine, its media readers and the signed policies it loads
    "vega.py",
    "video.py",
    "policy.py",
}
BACKEND_EXPORTS = {"VegaAgent"}
BACKEND_CALLABLES = {
    "uses_parallel_layout",
    "parallel_option_masks",
    "DecisionModel.__init__",
    "DecisionModel.forward",
    "build_model",
    "amp_dtype",
}
EXTENDED = {("agent.py", "Agent.__init__"), ("agent.py", "load"), ("router.py", "Router.__init__")}


def signature(node):
    value = copy.deepcopy(node.args)
    for arg in [
        *value.posonlyargs,
        *value.args,
        *value.kwonlyargs,
        *([value.vararg] if value.vararg else []),
        *([value.kwarg] if value.kwarg else []),
    ]:
        arg.annotation = None
    return value


def surface(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            result[node.name] = signature(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not method.name.startswith("_") or method.name == "__init__"
                ):
                    result[node.name + "." + method.name] = signature(method)
    # Core ML spells the unchanged default model through its hub constant.
    hub = path.parent / "hub.py"
    if hub.exists():
        for node in ast.parse(hub.read_text()).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MODEL" for t in node.targets)
            ):
                for sig in result.values():
                    sig.defaults = [
                        copy.deepcopy(node.value)
                        if isinstance(value, ast.Name) and value.id == "DEFAULT_MODEL"
                        else value
                        for value in sig.defaults
                    ]
    return result


def parameters(args):
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = {}
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        out[arg.arg] = (
            "posonly" if index < len(args.posonlyargs) else "positional",
            ast.dump(default) if default else "required",
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out[arg.arg] = ("keyword", ast.dump(default) if default else "required")
    return out


def exports(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", choices=["laya_mlx", "laya_coreml"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = root / ".upstream"
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PIN
    source = upstream / "laya"
    native = root / args.package
    results, failures = [], []
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source)
        if relative.parts[0] == "backends" or relative.name in BACKEND_MODULES:
            results.append(
                {
                    "module": str(relative),
                    "status": "backend_specific_module",
                    "reason": "PyTorch training, execution backend, compiler, ONNX or vegaml implementation; native runtime retained",
                }
            )
            continue
        target = native / ("router_cli.py" if relative == Path("cli.py") else relative)
        if not target.exists():
            failures.append("missing module: " + str(relative))
            continue
        expected, actual = surface(path), surface(target)
        if relative == Path("agent.py"):
            for name, value in surface(native / "runtime.py").items():
                if name.startswith("RuntimeMixin."):
                    actual.setdefault("Agent." + name.split(".", 1)[1], value)
        for name, sig in expected.items():
            row = {"module": str(relative), "callable": name}
            if relative == Path("common.py") and name in BACKEND_CALLABLES:
                row.update(
                    status="backend_tensor_boundary",
                    reason="Native tensor/model implementation replaces Torch tensor classes, masks and AMP dtype helpers",
                )
            elif name not in actual:
                failures.append(f"missing callable: {relative}:{name}")
                row["status"] = "missing"
            elif ast.dump(sig) == ast.dump(actual[name]):
                row["status"] = "signature_matches"
            elif (str(relative), name) in EXTENDED:
                want, have = parameters(sig), parameters(actual[name])
                different = {
                    key: [value, have.get(key)]
                    for key, value in want.items()
                    if have.get(key) != value
                }
                original_positions = [a.arg for a in [*sig.posonlyargs, *sig.args]]
                native_positions = [a.arg for a in [*actual[name].posonlyargs, *actual[name].args]]
                if native_positions[: len(original_positions)] != original_positions:
                    different["positional_order"] = [original_positions, native_positions]
                if different:
                    failures.append(f"changed upstream parameter: {relative}:{name}: {different}")
                row.update(
                    status="native_keyword_extensions",
                    additions={key: value for key, value in have.items() if key not in want},
                    changed=different,
                )
            else:
                failures.append(f"changed signature: {relative}:{name}")
                row["status"] = "changed"
            results.append(row)
    missing_exports = sorted(
        exports(source / "__init__.py") - exports(native / "__init__.py") - BACKEND_EXPORTS
    )
    if missing_exports:
        failures.append("missing root exports: " + str(missing_exports))
    report = {
        "upstream_revision": revision,
        "package": args.package,
        "scope": "public functions, explicitly defined public class methods, inherited Agent host methods and root __all__; annotations use native tensor types where needed",
        "rows": results,
        "missing_root_exports": missing_exports,
        "backend_specific_root_exports": sorted(BACKEND_EXPORTS),
        "failures": failures,
        "passed": not failures,
    }
    (root / "docs/UPSTREAM_API_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.package, len(results), "surface entries;", len(failures), "unclassified differences")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

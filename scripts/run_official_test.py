"""Run unchanged upstream host assertions against the MLX namespace.

Python imports and mock target module paths are rewritten. Router tests stub
Agent construction; dispatch that stub through the MLX load factory as well.
Host assertions, expected values and public model names are not rewritten.
Backend-only exclusions and the native MCP fixture translation are explicit below;
see docs/UPSTREAM.md for the complete scope.
"""

import ast
import re
import sys
import types
from pathlib import Path


class Namespace(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "laya":
            for alias in node.names:
                if alias.name == "cli":
                    alias.name, alias.asname = "router_cli", alias.asname or "cli"
        if node.module == "laya.cli":
            node.module = "laya.router_cli"
        if node.module == "laya" or (node.module or "").startswith("laya."):
            node.module = "laya_mlx" + node.module[4:]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "laya":
                alias.name, alias.asname = "laya_mlx", alias.asname or "laya"
            elif alias.name.startswith("laya."):
                # Unaliased imports need the original top-level binding too.
                if not alias.asname:
                    return [
                        ast.Import(names=[ast.alias(name="laya_mlx", asname="laya")]),
                        ast.Import(names=[ast.alias(name="laya_mlx" + alias.name[4:])]),
                    ]
                alias.name = "laya_mlx" + alias.name[4:]
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # This is a removal pattern for the original example's import;
            # rewriting it would stop the test's StubLaya from taking effect.
            if node.value == "\nimport laya\n":
                return node
            # The server logs under upstream's logger name; it is not a module path to rewrite.
            if node.value == "laya.serve":
                return node
            if "import laya" in node.value or "from laya" in node.value:
                node.value = re.sub(r"\bfrom laya(?=\b|\.)", "from laya_mlx", node.value)
                node.value = re.sub(
                    r"\bimport laya(?=[;\n ]|$)(?!\.)", "import laya_mlx as laya", node.value
                )
                node.value = node.value.replace(
                    "import laya.", "import laya_mlx as laya; import laya_mlx."
                )
                node.value = re.sub(r"([\"'])laya\.", r"\1laya_mlx.", node.value)
            if node.value.startswith("laya."):
                node.value = "laya_mlx" + node.value[4:]
                node.value = node.value.replace("laya_mlx.cli", "laya_mlx.router_cli")
        return node


def main():
    path = Path(sys.argv[1]).resolve()
    source = path.read_text(encoding="utf-8")
    if path.stem == "test_evals_api":
        # MLX refuses --onnx. Preserve Router revision assertions, excluding
        # only the ONNX runner setup and assertions that reference its reports.
        source = source.replace(
            "_real_router, _real_agent = laya.Router, _onnx_agent_module.ONNXAgent",
            "_real_router = laya.Router",
        ).replace(
            "laya.Router, _onnx_agent_module.ONNXAgent = _real_router, _real_agent",
            "laya.Router = _real_router",
        )
    if path.stem == "test_hooks":
        start = source.index(
            "# --------------------------------------------------------------- ONNXAgent parity"
        )
        end = source.index(
            "# --------------------------------------------------------------- BaseHook", start
        )
        source = source[:start] + source[end:]
    if path.stem == "test_confidence":
        # This host suite also embeds six PyTorch/TileLang padding implementation
        # and documentation checks. Neither backend is part of the MLX port.
        start = source.index("# ------------------------------------------- tl_kernels'")
        end = source.index("# --------------------------------------- DecisionResult's", start)
        source = source[:start] + source[end:]
    if path.stem == "test_mcp":
        # Retain all upstream cache assertions, translating only the fake
        # encoder/tokenizer/device to the native MLX tensor boundary.
        parsed = ast.parse(source)
        fixture = next(
            n
            for n in parsed.body
            if isinstance(n, ast.FunctionDef) and n.name == "test_shortlist_embed_cache"
        )
        fragment = ast.get_source_segment(source, fixture)
        fragment = fragment.replace("import torch", "import mlx.core as mx")
        fragment = fragment.replace("ids = input_ids.float()", "ids = input_ids.astype(mx.float32)")
        fragment = fragment.replace(
            "return _Out(torch.stack([ids, torch.ones_like(ids)], dim=-1))",
            "return mx.stack([ids, mx.ones_like(ids)], axis=-1)",
        )
        fragment = fragment.replace('torch.device("cpu")', "mx.cpu")
        start = fragment.index("    class CacheTok:")
        end = fragment.index("    def make_agent():", start)
        fragment = (
            fragment[:start]
            + """    class CacheTok:
        pad_token_id = 0

        def __init__(self):
            self.batches = []
            self.backend = self
            self.max_length = None

        def __deepcopy__(self, memo):
            clone = type(self)()
            clone.batches = self.batches
            return clone

        def enable_truncation(self, max_length):
            self.max_length = max_length

        def no_padding(self):
            pass

        def encode(self, text, add_special_tokens=True):
            self.batches.append([text])
            ids = [(ord(ch) % 5) + 1 for ch in text] or [1]
            return type("Encoded", (), {"ids": ids[:self.max_length]})()

"""
            + fragment[end:]
        )
        lines = source.splitlines(True)
        lines[fixture.lineno - 1 : fixture.end_lineno] = [fragment + "\n"]
        source = "".join(lines)
    tree = ast.parse(source)
    if path.stem == "test_hooks_api":

        class OnnxOnly(ast.NodeTransformer):
            @staticmethod
            def references(node):
                return any(
                    isinstance(n, ast.Name) and n.id in {"ONNXAgent", "_ApiONNXAgent", "VegaAgent"}
                    for n in ast.walk(node)
                )

            def visit_ImportFrom(self, node):
                return None if node.module in {"laya.onnx_agent", "laya.vega"} else node

            def visit_Tuple(self, node):
                node.elts = [e for e in node.elts if not self.references(e)]
                return self.generic_visit(node)

            def visit_List(self, node):
                node.elts = [e for e in node.elts if not self.references(e)]
                return self.generic_visit(node)

            def visit_Expr(self, node):
                # These APIs switch PyTorch forward implementations; MLX
                # rejects backend changes at construction (covered locally).
                backend = any(
                    isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "Agent"
                    and n.attr in {"set_backend", "backend", "backend_object"}
                    for n in ast.walk(node)
                )
                return None if self.references(node) or backend else node

        tree = OnnxOnly().visit(tree)
    if path.stem == "test_evals_api":

        class BackendOnly(ast.NodeTransformer):
            def visit_Import(self, node):
                return None if any(a.name == "laya.onnx_agent" for a in node.names) else node

            def visit_Assign(self, node):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if names & {"_onnx_agent_module", "_onnx_cfg", "_onnx_default_cfg"}:
                    return None
                return node

            def visit_Expr(self, node):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                return None if names & {"_onnx_cfg", "_onnx_default_cfg"} else node

        tree = BackendOnly().visit(tree)
    backend_tests = {
        "test_serve": {
            "test_health_without_a_resident_checkpoint_flags_the_preference",
            "test_health_agrees_with_the_mcp_status_tool",
            "test_build_router_strips_the_device_before_torch_sees_it",
            "test_thread_limit",
        },
        "test_mcp": {"test_device"},
        # These two build or import upstream's own vegaml engine; the lane's routing,
        # refusals and attached-agent behavior in the same files all still run.
        "test_router_vega": {
            "test_auto_turns_the_lane_on_when_vegaml_is_present_without_importing_it"
        },
        "test_langchain_native_policy": {
            "test_vega_policy_survives_the_adapter_without_model_weights"
        },
    }.get(path.stem, set())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.body = [
                m
                for m in node.body
                if not (isinstance(m, ast.FunctionDef) and m.name in backend_tests)
            ]
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in backend_tests)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in backend_tests
        )
        # A file that hands itself to pytest would collect upstream's own package from its
        # original path; the transformed tree is collected by pytest at the end instead.
        and not (
            isinstance(node, ast.If)
            and "__main__" in ast.unparse(node.test)
            and "pytest.main" in ast.unparse(node)
        )
    ]
    if path.stem == "test_cli":
        # MLX exports are inference-only. Identify the training command by
        # syntax, not line number, so upstream additions cannot hide other checks.
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "run_cli"
                and node.value.args
                and isinstance(node.value.args[0], ast.List)
                and all(isinstance(e, ast.Constant) for e in node.value.args[0].elts)
                and ast.literal_eval(node.value.args[0]) == ["train", "--help"]
            )
            and not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and str(node.value.args[0].value).startswith("train subcommand:")
            )
        ]
    if path.stem in {"test_structured", "test_structured_api", "test_confidence"}:
        # These host suites also inspect ONNX methods or run the ONNX decoder.
        # Exclude that backend's import/assertions; every host assertion is unchanged.
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "laya.onnx_agent")
            and not (
                isinstance(node, ast.Expr)
                and any(
                    isinstance(child, ast.Name) and child.id == "ONNXAgent"
                    for child in ast.walk(node)
                )
            )
        ]
    if path.stem in {"test_structured", "test_structured_api", "test_confidence"}:
        for node in ast.walk(tree):
            if isinstance(node, ast.Tuple):
                node.elts = [
                    e for e in node.elts if not (isinstance(e, ast.Name) and e.id == "ONNXAgent")
                ]
    tree = ast.fix_missing_locations(Namespace().visit(tree))
    if path.stem in {"test_router", "test_router_batch", "test_hooks"}:
        import laya_mlx.agent as agent

        agent.load = lambda *args, **kwargs: agent.Agent(*args, **kwargs)
    sys.argv = [str(path)]
    module = types.ModuleType("__main__")
    module.__file__ = str(path)
    sys.modules["__main__"] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    if path.stem == "_long_attribution":
        # Run the shared upstream attribution cases with its unchanged host fixture.
        # The fixture stubs encoding/forward/decoding, never native tensor execution.
        fixture_path = path.with_name("test_predict_long.py")
        fixture_tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
        fixture_tree.body = [
            node
            for node in fixture_tree.body
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"_Tok", "make_real_agent"}
            )
            or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "NO_STATE_STATS" for t in node.targets)
            )
        ]
        import threading

        import numpy as np

        from laya_mlx.agent import Agent

        fixture_scope = {"Agent": Agent, "np": np, "threading": threading}
        exec(compile(fixture_tree, str(fixture_path), "exec"), fixture_scope)
        checked = []

        def check(name, actual, expected):
            assert actual == expected, (name, actual, expected)
            checked.append(name)

        module.check_long_attribution(fixture_scope["make_real_agent"], check)
        print("%d unchanged upstream long-attribution checks passed" % len(checked))
    if any(name.startswith("test_") and callable(value) for name, value in module.__dict__.items()):
        import tempfile

        import pytest

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="test_mlx_", dir=path.parent
        ) as handle:
            handle.write(ast.unparse(tree))
            handle.flush()
            raise SystemExit(pytest.main(["-q", handle.name]))


if __name__ == "__main__":
    main()

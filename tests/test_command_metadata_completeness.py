import ast
from pathlib import Path


def _is_prefix_command_decorator(src: str, node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func_src = ast.get_source_segment(src, node.func) or ""
    return func_src.startswith("commands.command") or func_src.startswith("commands.hybrid_command")


def test_all_prefix_commands_have_help_metadata_fields() -> None:
    missing: list[str] = []

    for path in sorted(Path("commands").glob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue

            decorators = [d for d in node.decorator_list if _is_prefix_command_decorator(src, d)]
            if not decorators:
                continue

            command_name = node.name
            extras_obj: dict[str, object] | None = None
            for deco in decorators:
                assert isinstance(deco, ast.Call)
                for kw in deco.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        command_name = kw.value.value
                    if kw.arg == "extras":
                        extras_obj = ast.literal_eval(kw.value)

            if extras_obj is None:
                missing.append(f"{path}:{command_name}: missing extras")
                continue

            for key in ("destination", "plus", "pro"):
                value = extras_obj.get(key)
                if not isinstance(value, str) or not value.strip():
                    missing.append(f"{path}:{command_name}: missing extras.{key}")

    assert not missing, "\n".join(missing)

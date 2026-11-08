"""Validate and run LLM-generated decide() code.

Run as a separate process (python -I sandbox.py) with {code, items} JSON on
stdin; prints a JSON list of booleans. This is enough for local use only: it
has no OS-level isolation, so put it in a container before exposing the app.
"""
import ast
import json
import sys

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "round": round,
    "any": any, "all": all, "bool": bool, "int": int, "float": float, "str": str,
}
SAFE_METHODS = {"get", "lower", "upper", "strip", "startswith", "endswith"}
ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If,
    ast.IfExp, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Name,
    ast.Constant, ast.Attribute, ast.Call, ast.Subscript, ast.Assign, ast.Expr,
    ast.Tuple, ast.List, ast.Set, ast.Dict, ast.boolop, ast.operator,
    ast.unaryop, ast.cmpop, ast.expr_context,
)


def validate(code: str) -> None:
    """Raise ValueError unless code is exactly one safe decide(obj) function."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc}") from exc

    body = tree.body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "decide":
        raise ValueError("code must contain only: def decide(obj)")
    if body[0].decorator_list or len(body[0].args.args) != 1:
        raise ValueError("decide must take one argument and have no decorators")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not body[0]:
            raise ValueError("nested functions are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError(f"forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id not in SAFE_BUILTINS:
                raise ValueError(f"forbidden call: {target.id}")
            if not isinstance(target, (ast.Name, ast.Attribute)):
                raise ValueError("forbidden call target")


def _main() -> None:
    request = json.load(sys.stdin)
    validate(request["code"])
    scope = {"__builtins__": SAFE_BUILTINS}
    exec(request["code"], scope)  # validated above; restricted builtins
    decide = scope["decide"]

    results = []
    for item in request["items"]:
        try:
            results.append(bool(decide(item)))
        except Exception:
            results.append(True)  # privacy first: mask when the rule errors
    print(json.dumps(results))


if __name__ == "__main__":
    _main()

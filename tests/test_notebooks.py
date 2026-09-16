"""Keep supervision notebooks thin and free of retired experiment dependencies."""

import ast
import json
from pathlib import Path


def test_notebook_code_cells_have_valid_python() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


def test_benchmark_uses_core_without_retired_experimental_algorithms() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/02_reconstruction_benchmark.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)
    assert not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) for node in ast.walk(tree)
    )
    retired = {
        "assigned",
        "covered_area",
        "segment_roof_planes",
        "compute_local_features",
        "quality_acceptable",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint(retired)
    assert {
        "detect_roof_planes",
        "assess_lidar_quality",
        "evaluate_reconstruction_quality",
    } <= names

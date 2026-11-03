import ast
import importlib
import json
import unittest
from pathlib import Path

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].joinpath("notebooks").glob("*.ipynb"))


def imports(notebook: Path) -> list[tuple[str, list[str]]]:
    """(module, names imported from it) for every import in a notebook's code cells."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    # IPython magics and shell lines (% and !) are not Python.
    lines = [line for cell in cells if cell["cell_type"] == "code" for line in "".join(cell["source"]).splitlines()]
    tree = ast.parse("\n".join(line for line in lines if not line.lstrip().startswith(("%", "!"))))
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, [alias.name for alias in node.names]))
    return found


class NotebookImportTest(unittest.TestCase):
    def test_every_notebook_import_resolves(self) -> None:
        self.assertTrue(NOTEBOOKS, "no notebooks found")
        for notebook in NOTEBOOKS:
            for module, names in imports(notebook):
                with self.subTest(notebook=notebook.name, module=module):
                    imported = importlib.import_module(module)
                    for name in names:
                        if not hasattr(imported, name):
                            importlib.import_module(f"{module}.{name}")  # a submodule, e.g. from x import y


if __name__ == "__main__":
    unittest.main()

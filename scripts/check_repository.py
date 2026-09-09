"""Validate source syntax and notebook structure without executing notebooks."""

import ast
from pathlib import Path

import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py"))
    sources += sorted((root / "scripts").rglob("*.py"))
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not sources or not notebooks:
        raise RuntimeError("Expected Python source files and notebooks.")

    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

    print(f"Validated {len(sources)} Python files and {len(notebooks)} notebooks.")
    print("Notebooks were not executed; numerical results were not checked.")


if __name__ == "__main__":
    main()

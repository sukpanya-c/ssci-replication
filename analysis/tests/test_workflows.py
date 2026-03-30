from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.src.workflows import find_project_root


def test_workflows_module_supports_package_import() -> None:
    module = importlib.import_module("analysis.src.workflows")

    assert hasattr(module, "run_main_analysis_workflow")
    assert hasattr(module, "run_appendix_robustness_workflow")


def test_find_project_root_locates_ancestor(tmp_path: Path) -> None:
    project_root = tmp_path / "paper"
    nested = project_root / "analysis" / "notebooks"
    source_dir = project_root / "analysis" / "src"
    nested.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    assert find_project_root(nested) == project_root


def test_find_project_root_raises_when_analysis_src_is_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not locate project root"):
        find_project_root(tmp_path)

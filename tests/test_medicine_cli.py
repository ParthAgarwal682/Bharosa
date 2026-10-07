"""Medicine CLI load-then-search wiring.

The rows below are fixtures for the command boundary. They are not
production medicine records. ``load_medicines`` on this branch is the
call the command makes; these tests supply that function at the import
boundary and then let ``search_medicine`` rank the returned records.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import types
from pathlib import Path

import pytest

from bharosa.app.cli import _parser, main
from bharosa.medicine.search import search_medicine

_CLI_PATH = Path(__file__).resolve().parents[1] / "bharosa" / "app" / "cli.py"


def _row(brand: str) -> dict[str, object]:
    return {
        "brand": brand,
        "salt": "fixturesalta",
        "strength_value": 40,
        "strength_unit": "mg",
        "form": "tablet",
        "mrp": 12.5,
        "generic_price": None,
        "manufacturer": "fixture maker",
    }


def _use_loader(monkeypatch: pytest.MonkeyPatch, load_medicines: object) -> None:
    module = types.ModuleType("bharosa.medicine.loader")
    module.load_medicines = load_medicines  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bharosa.medicine.loader", module)


def _medicine_file_action() -> argparse.Action:
    parser = _parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    medicine = subparsers.choices["medicine"]
    return next(
        action for action in medicine._actions if "--medicine-file" in action.option_strings
    )


def test_medicine_file_option_is_required_without_a_default() -> None:
    action = _medicine_file_action()
    assert action.required is True
    assert action.default is None

    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda"])
    assert caught.value.code == 2


def test_missing_medicine_file_fails_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def load_medicines(path: Path) -> tuple[list[object], object]:
        nonlocal called
        called = True
        raise AssertionError(path)

    _use_loader(monkeypatch, load_medicines)
    missing = tmp_path / "missing.csv"

    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda", "--medicine-file", str(missing)])

    assert caught.value.code == 1
    assert called is False
    error = capsys.readouterr().err
    assert "medicine file not found:" in error
    assert "missing.csv" in error


def test_directory_medicine_path_fails_clearly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda", "--medicine-file", str(tmp_path)])

    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "medicine file is not a file:" in error


def test_loader_error_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "medicines.csv"
    dataset.write_text("not,a,medicine,file\n", encoding="utf-8")
    message = (
        "Dataset columns do not support agreed schema. "
        "Missing required columns: ['brand_name']"
    )

    def load_medicines(path: Path) -> tuple[list[object], object]:
        assert Path(path) == dataset
        raise ValueError(message)

    _use_loader(monkeypatch, load_medicines)

    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda", "--medicine-file", str(dataset)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "medicine load failed: ValueError: " in captured.err
    assert message in captured.err
    assert "matching search candidates" not in captured.out


def test_empty_records_do_not_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "medicines.csv"
    dataset.write_text("brand_name\n", encoding="utf-8")
    searched = False

    def load_medicines(path: Path) -> tuple[list[object], object]:
        assert Path(path) == dataset
        stats = types.SimpleNamespace(
            total_rows=4,
            accepted_rows=0,
            dropped_rows=4,
            duplicate_counts=1,
            reason_counts={"discontinued": 3, "duplicate_record": 1},
        )
        return [], stats

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal searched
        searched = True
        raise AssertionError((args, kwargs))

    _use_loader(monkeypatch, load_medicines)
    monkeypatch.setattr("bharosa.medicine.search.search_medicine", fail_if_called)

    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda", "--medicine-file", str(dataset)])

    assert caught.value.code == 1
    assert searched is False
    error = capsys.readouterr().err
    assert "produced no records" in error
    assert "Search was not run." in error
    assert "total_rows=4" in error
    assert "accepted_rows=0" in error


def test_loader_pair_is_unpacked_and_records_are_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "medicines.csv"
    dataset.write_text("brand\n", encoding="utf-8")
    rows = [_row("testbranda"), _row("testbrandb")]
    stats = types.SimpleNamespace(
        total_rows=3,
        accepted_rows=2,
        dropped_rows=1,
        duplicate_counts=0,
        reason_counts={"discontinued": 1},
    )
    seen: dict[str, object] = {}
    real_search = search_medicine

    def load_medicines(path: Path) -> tuple[list[dict[str, object]], object]:
        assert Path(path) == dataset
        seen["loaded_path"] = Path(path)
        return rows, stats

    def spy(query: str, k: int = 5, records: object = None) -> object:
        seen["query"] = query
        seen["k"] = k
        seen["records"] = records
        return real_search(query, k=k, records=records)  # type: ignore[arg-type]

    _use_loader(monkeypatch, load_medicines)
    monkeypatch.setattr("bharosa.medicine.search.search_medicine", spy)

    main(
        [
            "medicine",
            "testbranda",
            "--medicine-file",
            str(dataset),
            "-k",
            "1",
            "--verbose",
        ]
    )

    assert seen["loaded_path"] == dataset
    assert seen["query"] == "testbranda"
    assert seen["k"] == 1
    assert seen["records"] is rows
    output = capsys.readouterr().out
    assert "medicines.csv" in output
    assert "records: 2" in output
    assert "total_rows: 3" in output
    assert "accepted_rows: 2" in output
    assert "dropped_rows: 1" in output
    assert "duplicate_counts: 0" in output
    assert "discontinued" in output
    assert "brand=testbranda" in output
    assert "brand=testbrandb" not in output
    assert "fixturesalta" in output


def test_non_pair_loader_result_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "medicines.csv"
    dataset.write_text("brand\n", encoding="utf-8")

    def load_medicines(path: Path) -> list[dict[str, object]]:
        return [_row("testbranda"), _row("testbrandb")]

    _use_loader(monkeypatch, load_medicines)

    with pytest.raises(SystemExit) as caught:
        main(["medicine", "testbranda", "--medicine-file", str(dataset)])

    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "(records, stats)" in error
    assert "list" in error


def test_cli_source_has_no_hardcoded_dataset_path() -> None:
    source = _CLI_PATH.read_text(encoding="utf-8")
    assert "indian_pharmaceutical_products_clean.csv" not in source
    assert "data/medicines" not in source
    assert "data\\medicines" not in source
    assert "load_medicines(path)" in source
    assert "records, stats = loaded" in source
    assert "search_medicine(query, records=records, k=k)" in source
    assert "search_medicine(query, k=k)" not in source


def test_search_medicine_api_and_explicit_records_unchanged() -> None:
    signature = inspect.signature(search_medicine)
    assert list(signature.parameters) == ["query", "k", "records"]
    assert signature.parameters["k"].default == 5
    assert signature.parameters["records"].default is None

    rows = [
        _row("testbranda"),
        _row("otherbrandq"),
        {
            "brand": "xylophoneq",
            "salt": "fixturesaltb",
            "strength_value": 40,
            "strength_unit": "mg",
            "form": "tablet",
        },
    ]
    implied = search_medicine("testbranda", records=rows)
    explicit = search_medicine("testbranda", k=5, records=rows)

    assert [candidate.brand for candidate in implied.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    assert [candidate.brand for candidate in explicit.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    assert implied.candidates[0].salt == "fixturesalta"
    assert implied.candidates[0].strength_value == 40
    assert implied.format_result() == explicit.format_result()
